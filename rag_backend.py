import json
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("🚨 GROQ_API_KEY not found! Please check your .env file.")

# Initialize clients
client = Groq(api_key=api_key)
VECTOR_FILE = os.path.join("database", "rag_vectors.json")

def cosine_similarity(vec_a, vec_b):
    """Calculates the mathematical closeness of two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_chunks(query, top_k=3):
    """Embeds the query and finds the most relevant text chunks."""
    if not os.path.exists(VECTOR_FILE):
        return []
        
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        vectors_db = json.load(f)
        
    if not vectors_db:
        return []

    # Embed the user's question
    query_embedding = embedder.encode(query).tolist()

    # Compare query to every chunk in the DB
    results = []
    for item in vectors_db:
        sim = cosine_similarity(query_embedding, item["embedding"])
        results.append((sim, item["text_chunk"], item["file_link"]))
    
    # Sort by highest similarity and get the top K results
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def generate_rag_response(query):
    """Retrieves context and asks the LLM to answer based on that context."""
    retrieved_data = retrieve_top_chunks(query, top_k=3)
    
    if not retrieved_data:
        return "عذراً، قاعدة البيانات فارغة. يرجى رفع ملفات صوتية أولاً.", []

    # Build the context block
    context_text = "\n\n".join([f"- {item[1]} (المصدر: {item[2]})" for item in retrieved_data])
    sources = [item[2] for item in retrieved_data]

    prompt = f"""
أنت مساعد ذكي ومتخصص. أجب على سؤال المستخدم بناءً على "المعلومات المستخرجة" التالية فقط. 
إذا لم تكن الإجابة موجودة في المعلومات المستخرجة، قل "عذراً، لا أملك معلومات كافية للإجابة".

المعلومات المستخرجة:
{context_text}

سؤال المستخدم: {query}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip(), list(set(sources))