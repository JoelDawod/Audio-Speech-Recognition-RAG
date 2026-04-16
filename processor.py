import os
import math
import json
from groq import Groq
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("🚨 GROQ_API_KEY not found! Please check your .env file.")

# Initialize clients
client = Groq(api_key=api_key)
embedder = SentenceTransformer("BAAI/bge-m3", model_kwargs={"use_safetensors": True})

# Ensure database directory exists
DB_DIR = "database"
os.makedirs(DB_DIR, exist_ok=True)
META_FILE = os.path.join(DB_DIR, "rag_metadata.json")
VECTOR_FILE = os.path.join(DB_DIR, "rag_vectors.json")

# Initialize DB files if they don't exist
for db_file in [META_FILE, VECTOR_FILE]:
    if not os.path.exists(db_file):
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump([], f)

def get_audio_chunks(path, max_size_mb=20):
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb <= max_size_mb:
        return [path]
    
    audio = AudioSegment.from_file(path)
    duration_ms = len(audio)
    num_chunks = math.ceil(file_size_mb / max_size_mb)
    chunk_length_ms = duration_ms / num_chunks
    
    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_length_ms
        end = (i + 1) * chunk_length_ms
        chunk = audio[start:end]
        chunk_name = f"temp_chunk_{i}.mp3"
        chunk.export(chunk_name, format="mp3")
        chunk_paths.append(chunk_name)
    return chunk_paths

def transcribe_chunks(paths):
    full_text = ""
    for path in paths:
        with open(path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(path, file.read()),
                model="whisper-large-v3",
                language="ar",
                response_format="text"
            )
            full_text += transcription.strip() + " "
        if path.startswith("temp_chunk_"):
            os.remove(path)
    return full_text.strip()

def correct_and_title_text(raw_text):
    system_prompt = """
أنت خبير لغوي متخصص في تصحيح وتدقيق النصوص العربية الناتجة عن أنظمة التعرف على الكلام (ASR). 
مهمتك:
1. استنتاج عنوان مناسب جداً وقصير يعبر عن محتوى النص.
2. تصحيح الأخطاء الإملائية والنحوية بدقة.
يجب أن يكون مخرجك حصراً بهذا التنسيق:
العنوان: [ضع العنوان هنا]
النص المصحح: [ضع النص المصحح هنا]
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_text}],
        temperature=0.1
    )
    output = response.choices[0].message.content.strip()
    
    lines = output.split('\n')
    title, corrected_text = "بدون عنوان", output
    for i, line in enumerate(lines):
        if line.startswith("العنوان:"):
            title = line.replace("العنوان:", "").strip()
        elif line.startswith("النص المصحح:"):
            corrected_text = "\n".join(lines[i:]).replace("النص المصحح:", "").strip()
            break
    return title, corrected_text

def summarize_text(text):
    summary_prompt = """
أنت خبير في تحليل المحتوى وتلخيص النصوص العربية.
هيكلة الملخص:
1. "نظرة عامة": فقرة متماسكة.
2. "أهم التفاصيل": قائمة بالنقاط الجوهرية.
أخرج الملخص مباشرة بناءً على الهيكلة المطلوبة.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": summary_prompt}, {"role": "user", "content": text}],
        temperature=0.3, max_tokens=1024
    )
    return response.choices[0].message.content.strip()

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def process_and_vectorize_audio(audio_path):
    # 1. Pipeline Execution
    chunks = get_audio_chunks(audio_path)
    raw_transcript = transcribe_chunks(chunks)
    title, final_text = correct_and_title_text(raw_transcript)
    summary = summarize_text(final_text)
    
    metadata = {"file_link": audio_path, "title": title, "summary": summary, "full_text": final_text}
    text_chunks = chunk_text(final_text, chunk_size=100, overlap=20)
    
    new_vectors = []
    for chunk in text_chunks:
        embedding = embedder.encode(chunk).tolist()
        new_vectors.append({"file_link": audio_path, "text_chunk": chunk, "embedding": embedding})

    # 2. Append to Global Database
    with open(META_FILE, "r", encoding="utf-8") as f:
        all_meta = json.load(f)
    all_meta.append(metadata)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=4)

    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        all_vectors = json.load(f)
    all_vectors.extend(new_vectors)
    with open(VECTOR_FILE, "w", encoding="utf-8") as f:
        json.dump(all_vectors, f, ensure_ascii=False, indent=4)

    return metadata