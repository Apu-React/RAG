import chromadb
from sentence_transformers import SentenceTransformer
import ollama

CHROMA_DIR = "./chroma_store"
COLLECTION = "hr_policy"
MODEL_NAME  = "llama3"   # or "mistral", "phi3", etc.
TOP_K = 5                # how many chunks to retrieve

def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    return embed_model, collection

def retrieve(query, embed_model, collection, top_k=TOP_K):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]  # list of chunk strings

def build_prompt(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are an HR assistant. Answer the employee's question using ONLY the HR policy excerpts below.
If the answer is not in the excerpts, say "I don't have that information in the HR policy."

HR POLICY EXCERPTS:
{context}

EMPLOYEE QUESTION: {question}

ANSWER:"""

def ask(question, embed_model, collection):
    chunks = retrieve(question, embed_model, collection)
    prompt = build_prompt(question, chunks)
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def main():
    print("Loading models (first run may take a moment)...")
    embed_model, collection = load_resources()
    print("✅ HR Policy Bot ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = ask(question, embed_model, collection)
        print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    main()