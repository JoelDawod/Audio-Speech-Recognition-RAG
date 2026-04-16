import streamlit as st
import os
import json
from processor import process_and_vectorize_audio
from rag_backend import generate_rag_response

# --- UI Config ---
st.set_page_config(page_title="RAG Audio System", layout="wide")
st.title("🎙️ Arabic Audio RAG System")

os.makedirs("uploads", exist_ok=True)
META_FILE = os.path.join("database", "rag_metadata.json")

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Upload Audio")
    uploaded_file = st.file_uploader("Choose an MP3/WAV file", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(file_path)
        
        if st.button("Process & Add to Database"):
            with st.spinner("Processing audio, transcribing, and embedding... This may take a moment."):
                metadata = process_and_vectorize_audio(file_path)
                st.success(f"✅ Processed: {metadata['title']}")

# --- Main Area: Tabs ---
tab1, tab2 = st.tabs(["💬 RAG Chat Search", "📚 Document Library"])

# --- TAB 1: RAG Search ---
with tab1:
    st.header("Ask questions about your uploaded audio files")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("اسأل أي شيء عن الملفات الصوتية..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Searching database..."):
                answer, sources = generate_rag_response(prompt)
                
                # Format the response with sources
                full_response = f"{answer}\n\n"
                if sources:
                    full_response += f"**المصادر:** `{', '.join(sources)}`"
                
                st.markdown(full_response)
        
        # Save assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- TAB 2: Document Library ---
with tab2:
    st.header("Processed Audio Library")
    
    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
            
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['title']} ({doc['file_link']})"):
                    st.subheader("الملخص (Summary)")
                    st.write(doc['summary'])
                    st.subheader("النص الكامل (Full Text)")
                    st.write(doc['full_text'])
        else:
            st.info("No documents processed yet. Upload a file in the sidebar.")
    else:
        st.info("Database not initialized yet.")