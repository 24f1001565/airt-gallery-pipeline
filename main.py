from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import os
import requests
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime

load_dotenv()

app = FastAPI()

# Enable CORS
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
    global embeddings_db
    with open("embeddings.json", "r", encoding='utf-8') as f:
        embeddings_db = json.load(f)
    print(f"✅ Loaded {len(embeddings_db)} embeddings")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_query_embedding(query, api_key):
    """Get embedding for the query"""
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": query,
        "model": "text-embedding-3-small"
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()["data"][0]["embedding"]

def search_similar_chunks(query, top_k=5):  # Increased to 5
    """Search for most similar chunks"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Get query embedding
    query_embedding = get_query_embedding(query, api_key)
    
    # Calculate similarities
    similarities = []
    for item in embeddings_db:
        similarity = cosine_similarity(query_embedding, item["embedding"])
        similarities.append({
            "content": item["content"],
            "file": item["file"],
            "similarity": similarity
        })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return similarities[:top_k]

def generate_answer(query, context_chunks):
    """Generate answer using LLM API"""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Combine context
    context = "\n\n---DOCUMENT BREAK---\n\n".join([chunk["content"] for chunk in context_chunks])
    
    # STRICTER PROMPT
    prompt = f"""You are a documentation search assistant. Your ONLY job is to find and return the exact answer from the provided documentation.

STRICT RULES:
1. Read the documentation carefully
2. Find the EXACT answer to the question
3. Return ONLY the answer - a word, phrase, or short sentence
4. Do NOT add explanations, context, or extra words
5. Do NOT use backticks, quotes, or formatting unless they're part of the answer
6. If the answer is not clearly in the documentation, respond with exactly: NOT_FOUND

Documentation:
{context}

Question: {query}

Answer:"""
    
    # Call LLM API
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50,  # Force brevity
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    answer = response.json()["choices"][0]["message"]["content"].strip()
    
    # Clean up formatting
    answer = answer.strip('"').strip("'").strip("`").strip()
    
    return answer

@app.get("/search")
async def search(q: str = Query(..., description="Query string")):
    """Search endpoint with detailed logging"""
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*100)
    print(f"🔍 NEW REQUEST at {timestamp}")
    print("="*100)
    print(f"❓ QUESTION: {q}")
    print("-"*100)
    
    try:
        # Find similar chunks (increased to 5)
        similar_chunks = search_similar_chunks(q, top_k=5)
        
        # Print retrieved context with FULL CONTENT
        print("\n📚 RETRIEVED CONTEXT (FULL):")
        print("-"*100)
        for i, chunk in enumerate(similar_chunks, 1):
            print(f"\n{'='*20} [Chunk {i}] {'='*20}")
            print(f"📄 File: {chunk['file']}")
            print(f"📊 Similarity: {chunk['similarity']:.4f}")
            print(f"📏 Length: {len(chunk['content'])} characters")
            print(f"\n📝 FULL CONTENT:\n")
            print(chunk['content'])
            print(f"\n{'='*50}\n")
        
        # Generate answer
        print("\n🤖 Generating answer from LLM...")
        answer = generate_answer(q, similar_chunks)
        
        # Verify if answer exists in context
        print("\n🔍 VERIFICATION:")
        answer_found = False
        answer_lower = answer.lower().strip('`"\'')
        
        for i, chunk in enumerate(similar_chunks, 1):
            if answer_lower in chunk['content'].lower():
                print(f"  ✅ Answer '{answer}' FOUND in Chunk {i} ({chunk['file']})")
                answer_found = True
                
                # Show surrounding context
                content_lower = chunk['content'].lower()
                idx = content_lower.find(answer_lower)
                start = max(0, idx - 100)
                end = min(len(chunk['content']), idx + len(answer) + 100)
                context_snippet = chunk['content'][start:end]
                print(f"  📍 Context: ...{context_snippet}...")
                break
        
        if not answer_found and answer != "NOT_FOUND":
            print(f"  ⚠️  WARNING: Answer '{answer}' NOT FOUND in any retrieved chunk!")
            print(f"  🚨 This is likely a HALLUCINATION!")
        
        print("-"*100)
        print(f"✅ FINAL ANSWER: {answer}")
        print("-"*100)
        
        # Format sources
        sources = [
            {
                "file": chunk["file"],
                "similarity": float(chunk["similarity"])
            }
            for chunk in similar_chunks
        ]
        
        # Print sources summary
        print("\n📎 SOURCES SUMMARY:")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source['file']} (similarity: {source['similarity']:.4f})")
        
        print("="*100 + "\n")
        
        return {
            "answer": answer,
            "sources": sources,
            "context": [chunk["content"] for chunk in similar_chunks],  # Return full context
            "verified": answer_found
        }
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*100 + "\n")
        return {
            "error": str(e),
            "answer": "Error processing request"
        }

@app.get("/")
async def root():
    return {"message": "TypeScript RAG API", "endpoint": "/search?q=your_question"}

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    print(f"\n🌐 Incoming HTTP Request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"📤 Response Status: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

