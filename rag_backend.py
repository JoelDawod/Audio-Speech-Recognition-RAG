import json
import numpy as np
import os
from groq import Groq
from processor import embedder
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("🚨 GROQ_API_KEY not found! Please check your .env file.")

# Initialize clients
client = Groq(api_key=api_key)
VECTOR_FILE = os.path.join("database", "rag_vectors.json")

def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_chunks(query, top_k=3):
    if not os.path.exists(VECTOR_FILE):
        return []
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        vectors_db = json.load(f)
    if not vectors_db:
        return []

    query_embedding = embedder.encode(query).tolist()

    results = []
    for item in vectors_db:
        sim = cosine_similarity(query_embedding, item["embedding"])
        # We now return the text, the file link, AND the physical audio chunk
        results.append((sim, item["text_chunk"], item["file_link"], item["audio_chunk_path"]))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def generate_rag_response(query):
    retrieved_data = retrieve_top_chunks(query, top_k=3)
    
    if not retrieved_data:
        return "عذراً، قاعدة البيانات فارغة. يرجى رفع ملفات أولاً.", []

    # Build context for LLM
    context_text = "\n".join([f"- {item[1]}" for item in retrieved_data])
    
    # Save the audio chunks to send to the UI
    audio_sources = [{"source_file": item[2], "chunk_file": item[3], "text": item[1]} for item in retrieved_data]

    prompt = f"""
أنت مساعد ذكي. أجب على السؤال بناءً على المعلومات التالية فقط.
المعلومات المستخرجة: {context_text}
السؤال: {query}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip(), audio_sources