import json
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def get_embedding_openai(text, api_key):
    """Get embedding using OpenAI API"""
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()["data"][0]["embedding"]

def get_embedding_custom_api(text, api_url, api_key):
    """Get embedding using a custom API (like aipipe)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"text": text}  # Adjust based on your API's format
    
    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()["embedding"]  # Adjust based on response format

def generate_all_embeddings():
    """Generate embeddings for all chunks"""
    
    # Load chunks
    with open("chunks.json", "r", encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Get API credentials from environment
    api_key = "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"
    # api_url = os.getenv("EMBEDDING_API_URL")  # for custom API
    
    embeddings_data = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Use OpenAI or custom API
            embedding = get_embedding_openai(chunk["content"], api_key)
            # embedding = get_embedding_custom_api(chunk["content"], api_url, api_key)
            
            embeddings_data.append({
                "id": chunk["id"],
                "file": chunk["file"],
                "content": chunk["content"],
                "embedding": embedding
            })
            
            print(f"Processed {i+1}/{len(chunks)}")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing chunk {chunk['id']}: {e}")
    
    # Save embeddings
    with open("embeddings.json", "w", encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"Generated {len(embeddings_data)} embeddings")

if __name__ == "__main__":
    generate_all_embeddings()

