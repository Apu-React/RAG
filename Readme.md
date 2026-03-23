# HR Policy RAG Bot

A local RAG chatbot that answers questions from your HR policy PDF.

## Setup

1. Install dependencies:
```bash
   pip install -r requirements.txt
```

2. Install [Ollama](https://ollama.com) and pull a model:
```bash
   ollama pull tinyllama
```

3. Place your HR policy PDF in this folder and update `PDF_PATH` in `ingest.py`

4. Index the document:
```bash
   python ingest.py
```

5. Start the chatbot:
```bash
   python query.py
```

## Stack
- ChromaDB — local vector store
- sentence-transformers — embeddings
- Ollama — local LLM (no API key needed)