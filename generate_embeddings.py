import json
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def get_embedding(text, api_key):
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"input": text, "model": "text-embedding-3-small"}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def generate_all_embeddings():
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    api_key = os.getenv("OPENAI_API_KEY")
    embeddings_data = []

    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk["content"], api_key)
            embeddings_data.append({
                "id": chunk["id"],
                "file": chunk["file"],
                "content": chunk["content"],
                "embedding": embedding
            })
            print(f"✅ Processed {i+1}/{len(chunks)}")
            time.sleep(0.2)  # to avoid rate limiting
        except Exception as e:
            print(f"❌ Error processing {chunk['id']}: {e}")

    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=2)
    print(f"🎉 Generated {len(embeddings_data)} embeddings successfully!")

if __name__ == "__main__":
    generate_all_embeddings()
