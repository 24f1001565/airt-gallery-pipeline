import os
import json
import re
from pathlib import Path

def split_markdown_by_tokens(text, max_tokens=4096, overlap=200):
    """Split text into chunks by approximate token count"""
    # Rough approximation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        # Try to break at paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            break_point = text.rfind('\n\n', start, end)
            if break_point == -1:
                # Look for sentence break
                break_point = text.rfind('. ', start, end)
            if break_point != -1 and break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap_chars if end < len(text) else end
    
    return chunks

def process_markdown_files(repo_path):
    """Process all markdown files in the repository"""
    chunks_data = []
    chunk_id = 0
    
    # Find all markdown files
    md_files = list(Path(repo_path).rglob("*.md"))
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip empty files
            if not content.strip():
                continue
            
            # Split into chunks
            chunks = split_markdown_by_tokens(content)
            
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "id": f"{md_file.name}#chunk{i}",
                    "file": str(md_file.relative_to(repo_path)),
                    "content": chunk,
                    "chunk_index": i
                })
                chunk_id += 1
        
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
    
    return chunks_data

if __name__ == "__main__":
    chunks = process_markdown_files("typescript-book/docs")
    
    # Save to JSON file
    with open("chunks.json", "w", encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(chunks)} chunks")
