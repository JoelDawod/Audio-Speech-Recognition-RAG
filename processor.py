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

def extract_chunk_facts(chunk_text):
    prompt = f"""
أنت مساعد دقيق جداً. استخرج أهم المعلومات، الأرقام، والحقائق من هذا المقطع الصوتي القصير.
تجاهل الحشو والكلام الجانبي، وركز فقط على الجوهر.
إذا كان المقطع لا يحتوي على معلومات مهمة، اكتب "لا توجد معلومات جوهرية".

اكتب الإجابة في شكل نقاط (Bullet points) قصيرة ومباشرة:
{chunk_text}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def merge_extracted_facts(all_facts):
    prompt = f"""
أنت خبير في التلخيص المتقدم (NLP Summarization Expert). إليك قائمة عشوائية ومطولة من النقاط التي تم استخراجها تباعاً من تسجيل صوتي.

مهمتك هي "التوليف" (Synthesis) وليس مجرد الجمع. قم بدمج المعلومات المترابطة في نقاط كثيفة وغنية بالمعلومات، وتخلص من الحشو والمعلومات غير المهمة.

الشروط الصارمة:
1. الحد الأقصى: اكتب من 5 إلى 7 نقاط رئيسية كحد أقصى. لا تتجاوز 7 نقاط تحت أي ظرف.
2. الكثافة والدمج: ادمج الحقائق المترابطة في نقطة واحدة غنية. (مثال: بدلاً من كتابة وزن الحجر في نقطة، وارتفاع الهرم في نقطة، ادمج كل الأرقام الهندسية في نقطة واحدة تتحدث عن "الإعجاز الهندسي والأرقام").
3. الحذف: استبعد الأفكار المكررة، أو الاستنتاجات الضعيفة، أو الكلام العابر الذي لا يضيف قيمة علمية أو تاريخية.
4. استنتج "عنواناً" دقيقاً يعبر عن جوهر التسجيل بالكامل.

المخرجات يجب أن تكون بهذا التنسيق حصراً:
العنوان: [ضع العنوان هنا]
النقاط:
- [النقطة الشاملة الأولى]
- [النقطة الشاملة الثانية]

النقاط الخام المستخرجة:
{all_facts}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2 # Keep it low so the model respects the 7-point limit
    )
    output = response.choices[0].message.content.strip()
    
    # Parse the output
    lines = output.split('\n')
    title = "بدون عنوان"
    final_points = ""
    
    for i, line in enumerate(lines):
        if line.startswith("العنوان:"): 
            title = line.replace("العنوان:", "").strip()
        elif line.startswith("النقاط:"): 
            final_points = "\n".join(lines[i+1:]).strip()
            break
            
    # Fallback in case the model ignores the format
    if not final_points:
        final_points = output
        
    return title, final_points

def process_and_vectorize_audio(audio_path):
    print(f"Processing {audio_path}...")
    audio = AudioSegment.from_file(audio_path)
    base_name = os.path.basename(audio_path).split('.')[0]
    
    chunk_length_ms = 30 * 1000  # Slice audio into 30-second blocks
    full_raw_text = ""
    all_chunks_facts = ""
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
            
            chunk_facts = extract_chunk_facts(text)
            all_chunks_facts += f"\n- {chunk_facts}"

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

    print("Merging facts into final bullet points...")
    title, final_summary_points = merge_extracted_facts(all_chunks_facts)
    
    # We now pass `full_raw_text` directly! No LLM rewriting.
    metadata = {
        "file_link": audio_path, 
        "title": title, 
        "summary": final_summary_points, 
        "full_text": full_raw_text.strip() 
    }

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