import gradio as gr
import whisper
from gtts import gTTS
import groq
import requests
import tempfile
import os
import re
import nltk
import PyPDF2
import json
import io
from datetime import datetime, timedelta
from urllib.parse import urlparse
from docx import Document as DocxDocument

# RAG and ML Components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel
import torch

# Roboflow for plant disease detection
from inference_sdk import InferenceHTTPClient

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Initialize services using environment variables
GROQ_API_KEY = "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"
client = groq.Groq(api_key=GROQ_API_KEY)

ROBOFLOW_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="zL85GrIxufrU2bfcS2Sj"
)

WEATHER_API_KEY = "64be775708b8774817ae621feb910017"
WEATHER_URL = "https://api.openweathermap.org/data/2.5"

# Initialize services
print("🤖 Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded!")

print("🤖 Loading multilingual embedding model...")
multilingual_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("✅ Multilingual embeddings loaded!")

# PREDEFINED GOOGLE DRIVE PDF LINKS
PREDEFINED_PDF_LINKS = [
    "https://drive.google.com/file/d/1H7b-1PG2SLB99gjogfSl7QTOmLd1iGX0/view?usp=sharing",
    "https://drive.google.com/file/d/1Dzj18fyJXfS4_7hISa6vR1uDYibKne3m/view?usp=sharing",
    "https://drive.google.com/file/d/1xizmdkdG76SR7tJh3SnK6kviBLeA8KzX/view?usp=sharing",
    "https://drive.google.com/file/d/11x54pyujVtTerunzb-tcKrkaYKsLa3lb/view?usp=sharing",
    "https://drive.google.com/file/d/1jt5I5qSThcnYoGb2mD-zdRIUK-orV9UZ/view?usp=sharing",
    "https://drive.google.com/file/d/1M9_6NNUjd-TEpMmBF-Yoh85B7_Qd3FeK/view?usp=sharing",
    "https://drive.google.com/file/d/1HSwTjGxV5A5ThmIIXxZxGEZ--maOUJSd/view?usp=sharing",
    "",
]

class GoogleDrivePDFProcessor:
    """Handle Google Drive PDF extraction and processing"""

    @staticmethod
    def convert_gdrive_link(share_link):
        """Convert Google Drive share link to direct download link"""
        try:
            # Extract file ID from various Google Drive link formats
            patterns = [
                r'/file/d/([a-zA-Z0-9-_]+)',
                r'id=([a-zA-Z0-9-_]+)',
                r'/d/([a-zA-Z0-9-_]+)',
            ]

            file_id = None
            for pattern in patterns:
                match = re.search(pattern, share_link)
                if match:
                    file_id = match.group(1)
                    break

            if not file_id:
                return None

            # Create direct download link
            download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
            return download_link

        except Exception as e:
            print(f"Error converting Google Drive link: {e}")
            return None

    @staticmethod
    def download_pdf_from_gdrive(gdrive_link):
        """Download PDF from Google Drive link"""
        try:
            download_link = GoogleDrivePDFProcessor.convert_gdrive_link(gdrive_link)
            if not download_link:
                return None, "Invalid Google Drive link format"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(download_link, headers=headers, stream=True)

            # Handle Google Drive download confirmation
            if 'confirm=' in response.url:
                confirm_token = re.search(r'confirm=([^&]+)', response.url)
                if confirm_token:
                    confirmed_link = f"{download_link}&confirm={confirm_token.group(1)}"
                    response = requests.get(confirmed_link, headers=headers, stream=True)

            if response.status_code == 200:
                return response.content, "Success"
            else:
                return None, f"Download failed: Status {response.status_code}"

        except Exception as e:
            return None, f"Download error: {str(e)}"

    @staticmethod
    def extract_text_from_pdf(pdf_content):
        """Extract text from PDF content using PyPDF2"""
        try:
            # Create PDF reader from bytes
            pdf_stream = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)

            extracted_text = ""
            page_count = len(pdf_reader.pages)

            for page_num in range(page_count):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
                except Exception as page_error:
                    print(f"Error extracting page {page_num + 1}: {page_error}")
                    extracted_text += f"\n--- Page {page_num + 1} (Error) ---\n"

            return extracted_text, page_count

        except Exception as e:
            return f"PDF text extraction error: {str(e)}", 0

class AdvancedPakistaniAgriRAG:
    def __init__(self):
        self.embeddings = multilingual_embeddings
        self.vector_store = None
        self.numerical_data = {}
        self.gdrive_processor = GoogleDrivePDFProcessor()
        self.processed_documents = []  # Track processed documents
        self.setup_knowledge_base()
        # Automatically process predefined PDFs on initialization
        self.auto_process_predefined_pdfs()

    def process_mixed_language_text(self, text):
        """Process text that contains both English and Urdu"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        cleaned_text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\w\s.,;:!?()-]', ' ', text)
        return cleaned_text

    def extract_numerical_data(self, text):
        """Extract prices, yields, percentages from text"""
        numerical_info = {}

        prices = re.findall(r'[\$Rs\.]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
        if prices:
            numerical_info['prices'] = prices

        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        if percentages:
            numerical_info['percentages'] = percentages

        yields = re.findall(r'(\d+(?:\.\d+)?)\s*(tons?|kg|quintals?)\s*(?:per|/)?\s*(acre|hectare|ایکڑ)', text, re.IGNORECASE)
        if yields:
            numerical_info['yields'] = yields

        return numerical_info

    def setup_knowledge_base(self):
        """Create comprehensive Pakistani agricultural knowledge base"""

        pakistani_agri_knowledge = [
            {
                "content": """Punjab Wheat Varieties for Export:
                اعلیٰ قسم کی گندم کی اقسام:
                - Anmol-91: Yield 45-50 maunds/acre, Export price $320-350/ton
                - Faisalabad-2008: High protein 12-14%, Premium export variety
                - Galaxy-2013: Disease resistant, Suitable for UAE market
                - Punjab-2011: Good for bread making, Export to Afghanistan
                Urdu: یہ اقسام برآمد کے لیے بہترین ہیں اور زیادہ قیمت ملتی ہے""",
                "metadata": {"type": "crop_varieties", "region": "Punjab", "crop": "wheat", "language": "mixed"}
            },
            # Other knowledge entries...
        ]

        documents = []
        for item in pakistani_agri_knowledge:
            processed_content = self.process_mixed_language_text(item["content"])
            numerical_info = self.extract_numerical_data(processed_content)

            metadata = item["metadata"].copy()
            if numerical_info:
                metadata.update(numerical_info)

            doc = Document(page_content=processed_content, metadata=metadata)
            documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "۔", ".", ":", ";", " "],
            length_function=len
        )

        split_docs = text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

        print("✅ Advanced Pakistani Agricultural Knowledge Base Created!")
        print(f"📚 Loaded {len(split_docs)} knowledge chunks with mixed language support")

    def auto_process_predefined_pdfs(self):
        """Automatically process predefined Google Drive PDFs"""
        if not PREDEFINED_PDF_LINKS:
            print("📋 No predefined PDF links found")
            return

        print(f"🚀 Auto-processing {len(PREDEFINED_PDF_LINKS)} predefined PDF documents...")
        successfully_processed = 0

        for i, link in enumerate(PREDEFINED_PDF_LINKS, 1):
            try:
                print(f"📥 Processing document {i}/{len(PREDEFINED_PDF_LINKS)}...")

                # Download PDF
                pdf_content, download_status = self.gdrive_processor.download_pdf_from_gdrive(link)

                if pdf_content is None:
                    print(f"❌ Document {i}: {download_status}")
                    continue

                # Extract text using PyPDF2
                extracted_text, page_count = self.gdrive_processor.extract_text_from_pdf(pdf_content)

                if "error" in extracted_text.lower():
                    print(f"❌ Document {i}: {extracted_text}")
                    continue

                # Check if we got meaningful text
                if len(extracted_text.strip()) < 100:
                    print(f"⚠️ Document {i}: PDF may be image-based or encrypted. Limited text extracted.")

                # Process and add to knowledge base
                processed_text = self.process_mixed_language_text(extracted_text)
                numerical_info = self.extract_numerical_data(processed_text)

                # Create document with metadata
                doc = Document(
                    page_content=processed_text,
                    metadata={
                        "type": "auto_processed_pdf",
                        "source": f"Auto PDF {i}",
                        "pages": page_count,
                        "numerical_data": numerical_info,
                        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "original_link": link[:50] + "..." if len(link) > 50 else link
                    }
                )

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", "۔", ".", ":", ";", " "]
                )

                split_docs = text_splitter.split_documents([doc])

                # Add to vector store
                if self.vector_store:
                    self.vector_store.add_documents(split_docs)
                else:
                    self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

                # Track processed document
                self.processed_documents.append({
                    "id": i,
                    "pages": page_count,
                    "chunks": len(split_docs),
                    "source": link[:50] + "..." if len(link) > 50 else link,
                    "status": "✅ Success"
                })

                print(f"✅ Document {i}: {page_count} pages → {len(split_docs)} chunks")
                successfully_processed += 1

            except Exception as e:
                print(f"❌ Document {i}: Error - {str(e)}")
                self.processed_documents.append({
                    "id": i,
                    "pages": 0,
                    "chunks": 0,
                    "source": link[:50] + "..." if len(link) > 50 else link,
                    "status": f"❌ Error: {str(e)}"
                })

        print(f"🎉 Auto-processing complete! {successfully_processed}/{len(PREDEFINED_PDF_LINKS)} documents processed successfully")

    def get_knowledge_base_stats(self):
        """Get current knowledge base statistics"""
        total_chunks = sum(doc['chunks'] for doc in self.processed_documents if 'chunks' in doc)
        total_pages = sum(doc['pages'] for doc in self.processed_documents if 'pages' in doc)

        if not self.processed_documents:
            return "📊 Knowledge Base: Default Pakistani agricultural data only"

        stats = f"""📊 Knowledge Base Statistics:
🗂️ Auto-processed Documents: {len(self.processed_documents)}
📄 Total Pages Processed: {total_pages}
🧩 Total Text Chunks: {total_chunks}
📚 Default Knowledge: Pakistani agricultural data
🔍 Search Capability: Multilingual (English + Urdu)
✅ Status: Ready for queries
"""

        return stats

    def get_relevant_info(self, query, k=4):
        """Get relevant information with numerical data"""
        if not self.vector_store:
            return "Knowledge base not available"

        try:
            relevant_docs = self.vector_store.similarity_search(query, k=k)

            context = ""
            numerical_summary = ""

            for i, doc in enumerate(relevant_docs, 1):
                context += f"معلومات {i}: {doc.page_content}\n\n"

                metadata = doc.metadata
                if 'prices' in metadata:
                    numerical_summary += f"💰 قیمتیں: {', '.join(metadata['prices'])}\n"
                if 'percentages' in metadata:
                    numerical_summary += f"📊 فیصد: {', '.join(metadata['percentages'])}%\n"
                if 'yields' in metadata:
                    numerical_summary += f"🌾 پیداوار: {metadata['yields']}\n"

            if numerical_summary:
                context = f"📈 اہم اعداد و شمار:\n{numerical_summary}\n\n{context}"

            return context

        except Exception as e:
            return f"Error retrieving information: {str(e)}"

# Initialize advanced RAG system
print("🧠 Initializing Advanced Pakistani Agricultural Knowledge Base...")
pak_agri_rag = AdvancedPakistaniAgriRAG()

def voice_to_text(audio_file):
    """Convert farmer's voice to text with better error handling"""
    if audio_file is None:
        return ""

    try:
        result = whisper_model.transcribe(audio_file, language="ur")
        transcribed_text = result["text"]
        transcribed_text = pak_agri_rag.process_mixed_language_text(transcribed_text)
        return transcribed_text
    except Exception as e:
        return f"آواز سمجھ نہیں آئی: {str(e)}"

def get_enhanced_ai_response(user_message, location=""):
    """Get enhanced AI response with multilingual Pakistani knowledge"""

    relevant_context = pak_agri_rag.get_relevant_info(user_message)

    system_prompt = f"""
    آپ "زمین دوست" ہیں - پاکستانی کسانوں کے ماہر مشیر۔
    آپ کے پاس پاکستانی زراعت کی تازہ ترین معلومات ہیں (English اور Urdu میں):
    {relevant_context}
    کسان کا علاقہ: {location}
    آپ کا کام:
    1. پاکستانی حالات کے مطابق مشورہ دینا
    2. برآمدی فصلوں کی تجویز دینا (numerical data کے ساتھ)
    3. مقامی اقسام اور قیمتوں کا ذکر کرنا
    4. حکومتی اسکیموں کی معلومات دینا
    5. نقصان سے بچاؤ کے طریقے بتانا
    6. اعداد و شمار استعمال کرنا (prices, yields, percentages)
    Guidelines:
    - ہمیشہ "بھائی" کہہ کر شروع کریں
    - آسان اردو استعمال کریں
    - Numbers اور prices ضرور بتائیں
    - Export opportunities highlight کریں
    - Government schemes mention کریں
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama3-8b-8192",
            max_tokens=1200,
            temperature=0.7,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"معذرت، AI سے رابطہ نہیں ہو سکا: {str(e)}"

def get_weather_with_farming_advice(city="Lahore"):
    """Enhanced weather with numerical farming advice"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},PK&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        description = data['weather'][0]['description']

        farming_advice = ""
        if temp > 35:
            farming_advice = f"⚠️ زیادہ گرمی ({temp}°C): صبح 6-8 بجے پانی دیں، دوپہر میں نہیں۔ پانی کی مقدار 20% بڑھائیں۔"
        elif humidity > 80:
            farming_advice = f"🌧️ زیادہ نمی ({humidity}%): فنگیسائیڈ سپرے کریں۔ Mancozeb 2g/لیٹر یا Copper Oxychloride 3g/لیٹر۔"
        elif temp < 10:
            farming_advice = f"❄️ سردی ({temp}°C): پودوں کو ڈھانپیں، پانی 50% کم دیں۔ Frost protection ضروری۔"
        elif wind_speed > 5:
            farming_advice = f"💨 تیز ہوا ({wind_speed} m/s): کیڑے مار دوا کا سپرے نہ کریں۔ Wind barriers لگائیں۔"
        else:
            farming_advice = f"✅ موسم اچھا ہے ({temp}°C, {humidity}% نمی): کھیتی کے کام کر سکتے ہیں۔"

        return f"آج {city} میں {temp}°C، نمی {humidity}%، ہوا {wind_speed}m/s، موسم {description}\n\n{farming_advice}"

    except Exception as e:
        return f"موسمی معلومات نہیں مل سکیں: {str(e)}"

def text_to_voice(text):
    """Enhanced text to voice with better Urdu pronunciation"""
    try:
        clean_text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\w\s.,;:!?()-]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "... مکمل جواب اوپر پڑھیں"

        tts = gTTS(text=clean_text, lang='ur', slow=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def zameen_dost_advanced_chat(audio_input, text_input, city_name, focus_area):
    """Advanced main function with multilingual RAG and numerical data"""

    user_message = ""
    input_display = ""

    if audio_input is not None:
        user_message = voice_to_text(audio_input)
        input_display = f"💬 آپ نے کہا: {user_message}"
    elif text_input:
        user_message = text_input
        input_display = f"💬 آپ نے لکھا: {user_message}"

    if not user_message.strip():
        return "کرپیا کوئی سوال پوچھیں", None, "❌ کوئی سوال نہیں ملا"

    enhanced_message = user_message
    if focus_area != "عام سوال":
        enhanced_message += f" (کسان کی دلچسپی: {focus_area})"

    if any(word in user_message for word in ["موسم", "بارش", "پانی", "weather", "irrigation", "spray", "سپرے"]):
        weather_info = get_weather_with_farming_advice(city_name)
        enhanced_message += f"\n\nموسمی حالات: {weather_info}"

    ai_response = get_enhanced_ai_response(enhanced_message, city_name)
    voice_response = text_to_voice(ai_response)

    return input_display, voice_response, ai_response

# Plant disease detection function
def predict_disease(image_path):
    try:
        result = ROBOFLOW_CLIENT.infer(image_path, model_id="plant-disease-detection-v2-2nclk/1")
        if not result.get("predictions"):
            return "❌ کوئی بیماری معلوم نہیں ہوئی"
        pred = result["predictions"][0]
        return f"🦠 بیماری: {pred['class']}\nاعتماد: {pred['confidence']*100:.2f}%"
    except Exception as e:
        return f"❌ Roboflow Error: {e}"

# Weather functions
def get_current_weather(city):
    try:
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "ur"}
        res = requests.get(f"{WEATHER_URL}/weather", params=params).json()
        if res.get("cod") != 200:
            return "❌ شہر کا نام درست نہیں"
        return (
            f"🌤️ موسم: {res['weather'][0]['description']}\n"
            f"🌡️ درجہ حرارت: {res['main']['temp']}°C (محسوس: {res['main']['feels_like']}°C)\n"
            f"💧 نمی: {res['main']['humidity']}%"
        )
    except Exception as e:
        return f"❌ Weather API Error: {e}"

def get_tomorrow_weather(city):
    try:
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "ur"}
        res = requests.get(f"{WEATHER_URL}/forecast", params=params).json()
        if res.get("cod") != "200":
            return "❌ شہر کا نام درست نہیں"

        tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
        for entry in res["list"]:
            dt_txt = entry["dt_txt"]
            if str(tomorrow) in dt_txt and "12:00:00" in dt_txt:
                description = entry["weather"][0]["description"]
                return (
                    f"🗓️ کل کا موسم: {description}\n"
                    f"🌡️ درجہ حرارت: {entry['main']['temp']}°C\n"
                    f"💧 نمی: {entry['main']['humidity']}%"
                )
        return "❌ کل کی پیشگوئی نہیں ملی"
    except Exception as e:
        return f"❌ API Error: {e}"

# Create the enhanced app with lighter UI
with gr.Blocks(
    title="Smart Zameen Dost - زمین دوست",
    theme=gr.themes.Base(),
    css="""
    .gradio-container {
        background: linear-gradient(135deg, #f8fdff 0%, #e8f7f8 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #2E8B57;
    }
    .stats-box {
        background: linear-gradient(45deg, #e8f5e8, #f0f8e8);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c8e6c9;
        margin: 10px 0;
        font-size: 0.9em;
    }
    """
) as app:

    # Clean and simple header
    gr.HTML("""
    <div class='header-box'>
        <div style='text-align: center;'>
            <h1 style='color: #2E8B57; font-size: 2.2em; margin: 0 0 8px 0;'>🌾 Smart Zameen Dost</h1>
            <p style='color: #666; font-size: 1.1em; margin: 0;'>پاکستانی کسانوں کا ذہین مشیر</p>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 اپنا سوال پوچھیں")

            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="آواز میں پوچھیں"
            )

            text_input = gr.Textbox(
                label="یا یہاں لکھیں (اردو/English)",
                placeholder="مثال: کون سی فصل زیادہ منافع دے گی؟",
                lines=2
            )

            with gr.Row():
                city_input = gr.Textbox(
                    label="آپ کا شہر",
                    placeholder="Lahore, Karachi, Faisalabad",
                    value="Lahore",
                    scale=1
                )

                focus_area = gr.Dropdown(
                    label="دلچسپی کا شعبہ",
                    choices=[
                        "عام سوال",
                        "برآمدی فصلیں",
                        "گندم کی کاشت",
                        "چاول کی کاشت",
                        "کپاس کی کاشت",
                        "سبزیوں کی کاشت",
                        "پھلوں کی کاشت",
                        "کھاد اور بیج",
                        "بیماریوں کا علاج",
                        "حکومتی اسکیمز",
                        "منڈی کی قیمتیں"
                    ],
                    value="عام سوال",
                    scale=1
                )

            chat_btn = gr.Button("🚀 جواب حاصل کریں", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 ذہین جواب")

            input_display = gr.Textbox(
                label="آپ کا سوال",
                lines=2,
                interactive=False
            )

            audio_output = gr.Audio(
                label="🔊 آواز میں جواب"
            )

            text_output = gr.Textbox(
                label="📝 تفصیلی جواب",
                lines=10,
                interactive=False,
                show_copy_button=True
            )

    # Knowledge base statistics
    with gr.Row():
        kb_stats = gr.HTML(
            value=pak_agri_rag.get_knowledge_base_stats(),
            elem_classes=["stats-box"]
        )

    # Connect chat function
    chat_btn.click(
        zameen_dost_advanced_chat,
        inputs=[audio_input, text_input, city_input, focus_area],
        outputs=[input_display, audio_output, text_output]
    )

    with gr.Tab("🌿 پودوں کی بیماری"):
        img_input = gr.Image(type="filepath", label="📷 پتے کی تصویر")
        disease_output = gr.Textbox(label="🔎 بیماری کی تفصیل")
        gr.Button("🔍 شناخت کریں").click(fn=predict_disease, inputs=img_input, outputs=disease_output)

    with gr.Tab("🌦️ آج کا موسم"):
        city_now = gr.Textbox(label="🏙️ شہر کا نام")
        weather_now = gr.Textbox(label="☁️ موسم")
        gr.Button("موسم دکھائیں").click(fn=get_current_weather, inputs=city_now, outputs=weather_now)

    with gr.Tab("📅 کل کا موسم"):
        city_tomorrow = gr.Textbox(label="🏙️ شہر کا نام")
        weather_tomorrow = gr.Textbox(label="🌧️ موسم کی تفصیل")
        gr.Button("کل کا موسم دیکھیں").click(fn=get_tomorrow_weather, inputs=city_tomorrow, outputs=weather_tomorrow)

    # Simple footer
    gr.HTML("""
    <div style='text-align: center; padding: 15px; margin-top: 20px; background: rgba(46,139,87,0.1); border-radius: 10px;'>
        <p style='color: #2E8B57; margin: 0; font-size: 1em; font-weight: bold;'>Smart Zameen Dost - Made for Pakistani Farmers 🇵🇰</p>
    </div>
    """)


print("🎉 Smart Zameen Dost with Auto PDF Processing is ready!")
print(f"\n✅ Auto-processed {len(PREDEFINED_PDF_LINKS)} PDF documents from Google Drive")
print("📊 Knowledge base includes default Pakistani agricultural data + your PDFs")
print("🔍 Multilingual search capability (English + Urdu)")
print("🎤 Voice input and output support")
print("🌡️ Weather-based farming advice")
print("💰 Export market data with prices")
print("\n📋 To add more PDFs:")
print("  1. Add Google Drive share links to PREDEFINED_PDF_LINKS list")
print("  2. Restart the application")
print("  3. PDFs will be automatically processed and added to knowledge base")
print("\n🚀 Application is ready to serve Pakistani farmers!")

# Launch the enhanced app
if __name__ == "__main__":
    app.launch(
        share=True,
        debug=True,
        show_error=True
    )
