import streamlit as st
import os
import json
from groq import Groq
from processor import process_and_vectorize_audio
from rag_backend import generate_rag_response

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("🚨 GROQ_API_KEY not found! Please check your .env file.")

st.set_page_config(page_title="Multimodal RAG System", layout="wide")
st.title("🎙️ Multimodal Audio RAG System")

os.makedirs("uploads", exist_ok=True)
META_FILE = os.path.join("database", "rag_metadata.json")

client = Groq(api_key=api_key)

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload Audio to Library")
    uploaded_file = st.file_uploader("Upload an MP3/WAV", type=["mp3", "wav"])
    if uploaded_file and st.button("Process & Index"):
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Slicing audio, transcribing, and embedding..."):
            metadata = process_and_vectorize_audio(file_path)
            st.success(f"✅ Indexed: {metadata['title']}")

# --- Main Area ---
tab1, tab2 = st.tabs(["💬 Voice & Text Search", "📚 Document Library"])

with tab1:
    st.header("Search your Audio Database")
    
    # 1. Input Methods (Text or Microphone)
    col1, col2 = st.columns([1, 1])
    with col1:
        text_query = st.chat_input("Type your question here...")
    with col2:
        mic_query = st.audio_input("🎤 Or record your question via Microphone")

    final_query = None

    # Determine which input was used
    if text_query:
        final_query = text_query
    elif mic_query:
        with st.spinner("Transcribing your voice..."):
            # Save mic audio temp file
            with open("temp_mic.wav", "wb") as f:
                f.write(mic_query.getbuffer())
            # Transcribe the mic audio using Whisper
            with open("temp_mic.wav", "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=("temp_mic.wav", file.read()),
                    model="whisper-large-v3", language="ar", response_format="text"
                )
            final_query = transcription.strip()
            st.info(f"🗣️ You asked: **{final_query}**")

    # 2. Process the Query
    if final_query:
        with st.spinner("Searching audio embeddings..."):
            answer, audio_sources = generate_rag_response(final_query)
            
            st.markdown("### 🤖 Answer:")
            st.write(answer)
            
            if audio_sources:
                st.markdown("### 🎧 Audio Evidence (Top Matches):")
                for i, source in enumerate(audio_sources):
                    with st.expander(f"🔊 Listen to Match {i+1} (From: {os.path.basename(source['source_file'])})"):
                        st.audio(source["chunk_file"])
                        st.write(f"**Transcript:** {source['text']}")

with tab2:
    st.header("Processed Audio Library")
    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
        for doc in documents:
            with st.expander(f"📄 {doc['title']}"):
                st.write("**الملخص:**", doc['summary'])
                st.write("**النص:**", doc['full_text'])
    else:
        st.info("Library is empty.")