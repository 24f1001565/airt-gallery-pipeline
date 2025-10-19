from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import os
import requests
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime

# Load environment variables (API keys, etc.)
load_dotenv()

app = FastAPI(title="TypeScript RAG API", version="1.0")

# Enable CORS (allow all origins for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings at startup
embeddings_db = []


@app.on_event("startup")
async def load_embeddings():
    """Load precomputed embeddings from embeddings.json"""
    global embeddings_db
    if not os.path.exists("embeddings.json"):
        raise FileNotFoundError("❌ embeddings.json not found. Run generate_embeddings.py first.")
    with open("embeddings.json", "r", encoding="utf-8") as f:
        embeddings_db = json.load(f)
    print(f"✅ Loaded {len(embeddings_db)} embeddings into memory.")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_query_embedding(query, api_key):
    """Get embedding vector for query using AIPipe endpoint."""
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"input": query, "model": "text-embedding-3-small"}

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def search_similar_chunks(query, top_k=5):
    """Return top-k most similar chunks for a given query."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing OPENAI_API_KEY in .env")

    query_embedding = get_query_embedding(query, api_key)
    similarities = []

    for item in embeddings_db:
        similarity = cosine_similarity(query_embedding, item["embedding"])
        similarities.append(
            {"content": item["content"], "file": item["file"], "similarity": similarity}
        )

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


def generate_answer(query, context_chunks):
    """Use LLM (via AIPipe) to extract the exact answer from the documentation."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    context = "\n\n---DOCUMENT BREAK---\n\n".join(
        [chunk["content"] for chunk in context_chunks]
    )

    prompt = f"""
You are a strict documentation-based QA assistant.
Your task: Find the EXACT answer to the question from the given documentation.
Rules:
1. Return only the exact answer (word, phrase, or short sentence).
2. No explanations or rephrasing.
3. If not found, reply exactly: NOT_FOUND.

Documentation:
{context}

Question: {query}

Answer:
"""

    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()

    return answer.strip('`"\' ')


@app.get("/search")
async def search(q: str = Query(..., description="Search query text")):
    """Main search endpoint"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80)
    print(f"🔍 Query at {timestamp}: {q}")
    print("=" * 80)

    try:
        similar_chunks = search_similar_chunks(q, top_k=5)
        print(f"📚 Retrieved {len(similar_chunks)} relevant chunks.")

        for i, c in enumerate(similar_chunks, 1):
            print(f"\n[{i}] {c['file']} (sim={c['similarity']:.4f})")
            print(c["content"][:300] + "..." if len(c["content"]) > 300 else c["content"])

        print("\n🤖 Sending to LLM...")
        answer = generate_answer(q, similar_chunks)

        # Verify correctness
        found = any(answer.lower() in c["content"].lower() for c in similar_chunks)

        print("\n✅ Final Answer:", answer)
        if not found and answer != "NOT_FOUND":
            print("⚠️ Warning: Possible hallucination (answer not found in context)")

        return {
            "answer": answer,
            "verified": found,
            "sources": [
                {"file": c["file"], "similarity": float(c["similarity"])}
                for c in similar_chunks
            ],
            "context": [c["content"] for c in similar_chunks],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "answer": "Error processing request"}


@app.get("/")
async def root():
    return {
        "message": "TypeScript RAG API running!",
        "usage": "/search?q=your_question",
    }


@app.middleware("http")
async def log_requests(request, call_next):
    print(f"\n🌐 {request.method} {request.url}")
    response = await call_next(request)
    print(f"📤 Status: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
