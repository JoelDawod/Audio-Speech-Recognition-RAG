import os
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

DB_DIR = "database"
AUDIO_DIR = os.path.join(DB_DIR, "audio_chunks")
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

META_FILE = os.path.join(DB_DIR, "rag_metadata.json")
VECTOR_FILE = os.path.join(DB_DIR, "rag_vectors.json")

for db_file in [META_FILE, VECTOR_FILE]:
    if not os.path.exists(db_file):
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump([], f)

def correct_and_summarize(raw_text):
    prompt = f"""
أنت خبير لغوي. قم بتصحيح النص التالي إملائياً ونحوياً، واستنتج له عنواناً، واكتب ملخصاً قصيراً له.
أخرج الإجابة حصراً بهذا التنسيق:
العنوان: [العنوان هنا]
الملخص: [الملخص هنا]
النص المصحح: [النص هنا]

النص الأصلي: {raw_text}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    output = response.choices[0].message.content.strip()
    
    lines = output.split('\n')
    title, summary, corrected_text = "بدون عنوان", "", output
    for i, line in enumerate(lines):
        if line.startswith("العنوان:"): title = line.replace("العنوان:", "").strip()
        elif line.startswith("الملخص:"): summary = line.replace("الملخص:", "").strip()
        elif line.startswith("النص المصحح:"): 
            corrected_text = "\n".join(lines[i:]).replace("النص المصحح:", "").strip()
            break
    return title, summary, corrected_text

def process_and_vectorize_audio(audio_path):
    print(f"Processing {audio_path}...")
    audio = AudioSegment.from_file(audio_path)
    base_name = os.path.basename(audio_path).split('.')[0]
    
    chunk_length_ms = 30 * 1000  # Slice audio into 30-second blocks
    full_raw_text = ""
    new_vectors = []
    
    # 1. Slice audio, transcribe, and embed each physical chunk
    for i in range(0, len(audio), chunk_length_ms):
        start_ms = i
        end_ms = min(i + chunk_length_ms, len(audio))
        chunk_audio = audio[start_ms:end_ms]
        
        # Save the physical audio chunk
        chunk_filename = f"{base_name}_{start_ms}_{end_ms}.mp3"
        chunk_path = os.path.join(AUDIO_DIR, chunk_filename)
        chunk_audio.export(chunk_path, format="mp3")
        
        # Transcribe this specific chunk
        with open(chunk_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(chunk_path, file.read()),
                model="whisper-large-v3",
                language="ar",
                response_format="text"
            )
        
        text = transcription.strip()
        if text:
            full_raw_text += text + " "
            # Embed the text of this specific audio chunk
            embedding = embedder.encode(text).tolist()
            new_vectors.append({
                "file_link": audio_path,
                "audio_chunk_path": chunk_path,
                "start_time": start_ms / 1000.0,
                "end_time": end_ms / 1000.0,
                "text_chunk": text,
                "embedding": embedding
            })

    # 2. Correct and summarize the full text for the library view
    title, summary, final_text = correct_and_summarize(full_raw_text)
    metadata = {"file_link": audio_path, "title": title, "summary": summary, "full_text": final_text}

    # 3. Save to Databases
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