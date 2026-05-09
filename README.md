# 🎙️ Multimodal Audio RAG System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Groq](https://img.shields.io/badge/Groq-Fast_Inference-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![AI Engineering](https://img.shields.io/badge/AI_Engineering-RAG_Pipeline-success.svg)

An advanced **Retrieval-Augmented Generation (RAG)** pipeline that processes, embeds, and queries audio files. Instead of relying on raw audio embeddings, this system utilizes an **Audio-Linked Text RAG** architecture—embedding highly accurate transcripts while returning the exact, playable audio slices as evidence alongside the LLM's answers.

## 🚀 Key Features

* **Audio-Linked Semantic Search:** Slices audio into 30-second chunks, transcribes them, and embeds the text. When queried, it returns the LLM-generated answer alongside the exact playable `.mp3` chunks where the information was found.
* **Map-Reduce Summarization:** Overcomes LLM context limits and "lazy synthesis" by extracting dense facts from individual audio slices, then synthesizing them into a strict, executive-level 5-7 point summary.
* **Multimodal Input:** Supports both standard text queries and **Voice Search** via microphone, utilizing real-time Whisper transcription to query the vector database.
* **Arabic NLP Optimization:** Specifically tailored to handle complex historical and religious Arabic texts, correcting phonetic ASR errors contextually before embedding.
* **Interactive UI:** A complete Streamlit dashboard featuring a Chat interface, audio players for evidence retrieval, and a Document Library.

## 🧠 System Architecture & Pipeline

### 1. Ingestion & Processing (`processor.py`)
1. **Audio Slicing:** Large audio files are physically sliced into 30-second `mp3` chunks using `pydub`.
2. **Speech-to-Text (ASR):** Chunks are transcribed using **Whisper-Large-V3** via Groq's high-speed API.
3. **Map-Reduce Synthesis:** * *Map:* Facts are extracted from every individual 30-second transcript.
   * *Reduce:* A prompt-engineered LLM merges the facts into a highly dense, non-repetitive bulleted summary.
4. **Vectorization:** The chunked text is embedded using `BAAI/bge-m3` (optimized for multilingual/Arabic semantic search).
5. **Metadata Linking:** Embeddings are saved alongside their specific timestamps and physical audio file paths.

### 2. Retrieval & Generation (`rag_backend.py`)
1. **Query Embedding:** User queries (text or transcribed voice) are embedded.
2. **Cosine Similarity Search:** The system calculates the mathematical distance between the query vector and the document vectors using `numpy`.
3. **Context Injection:** The top-K retrieved text chunks are injected into the LLM prompt.
4. **Evidence Return:** The LLM generates a grounded answer, and the backend passes the linked physical audio chunks back to the frontend.

### 3. User Interface (`app.py`)
Built with Streamlit, the UI allows users to upload `.mp3`/`.wav` files, view processed summaries/transcripts, and chat with their audio database using text or voice.

## 🛠️ Technology Stack

* **LLM Orchestration:** `llama-3.3-70b-versatile` (via Groq API)
* **Speech-to-Text:** `whisper-large-v3` (via Groq API)
* **Embedding Model:** `BAAI/bge-m3` (via `sentence-transformers`)
* **Audio Processing:** `pydub`, `ffmpeg`
* **Vector Math:** `numpy`
* **Frontend:** `streamlit`

## ⚙️ Installation & Setup

### Prerequisites
1. **Python 3.9+** installed on your machine.
2. **FFmpeg** installed (Required by `pydub` for audio slicing).
   * *Windows:* `winget install ffmpeg`
   * *Mac:* `brew install ffmpeg`
   * *Linux:* `sudo apt install ffmpeg`
3. A **Groq API Key** (Free tier available at console.groq.com).

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/multimodal-audio-rag.git
cd multimodal-audio-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

## 💻 How to Run

Launch the Streamlit application by running the following command in your terminal:

```bash
streamlit run app.py
```

1. Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
2. Upload an audio file via the sidebar and click **Process & Index**.
3. Once processed, use the **Chat Tab** to ask questions (via text or mic) or view the generated notes in the **Document Library Tab**.

## 🔮 Future Enhancements
* Migrate the local JSON vector storage to a dedicated Vector DB (e.g., ChromaDB, FAISS, Qdrant) for massive-scale production.
* Add source-highlighting in the UI to auto-scroll to the exact sentence in the Document Library.