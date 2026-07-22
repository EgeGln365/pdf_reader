# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions based only on the document content.

The project is built with **LangChain**, **FAISS**, **OpenAI**, and **Gradio**, following a modular architecture for easy extension and experimentation.

---

## Features

- 📄 Upload any PDF document
- ✂️ Automatic document chunking
- 🧠 Embedding generation using HuggingFace BGE
- 🔍 Semantic retrieval with FAISS
- 🤖 OpenAI-powered question answering
- 📚 Source page references
- 🖥️ Simple Gradio web interface

---

## Project Structure

```text
pdf_reader/
│
├── app.py
├── requirements.txt
├── .env
│
├── app/
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── llm.py
│   ├── prompt.py
│   └── rag.py
│
└── sample_pdfs/
```

---

## Tech Stack

- Python
- LangChain
- OpenAI
- HuggingFace Embeddings (BGE)
- FAISS
- Gradio

---

## How It Works

```text
PDF Upload
      │
      ▼
Load PDF
      │
      ▼
Split into Chunks
      │
      ▼
Generate Embeddings
      │
      ▼
Create FAISS Vector Store
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Build Prompt
      │
      ▼
OpenAI LLM
      │
      ▼
Answer + Sources
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/EgeGln365/pdf_reader.git
cd pdf_reader
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

Run the application:

```bash
python app.py
```

---

## Future Improvements

- Chat history support
- Multiple PDF upload
- Persistent FAISS index
- Streaming responses
- Hybrid Search (BM25 + FAISS)
- Reranker integration
- RAG evaluation pipeline
- LangGraph agent workflow

---

## License

This project is for educational and portfolio purposes.