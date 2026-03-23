import fitz  # pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
import os


PDF_PATH = r"C:\Users\Admin\OneDrive\hr_rag\hr_policy.pdf"
CHROMA_DIR = "./chroma_store"
COLLECTION = "hr_policy"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def ingest():
    print("📄 Extracting text from PDF...")
    text = extract_text(PDF_PATH)
    chunks = chunk_text(text)
    print(f"✂️  Created {len(chunks)} chunks")

    print("🔢 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, free, local

    print("📐 Embedding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    print("💾 Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if re-ingesting
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"✅ Done! {len(chunks)} chunks stored in {CHROMA_DIR}")

if __name__ == "__main__":
    ingest()