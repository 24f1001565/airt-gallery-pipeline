import os
import json
from pathlib import Path

def split_markdown_by_tokens(text, max_tokens=4096, overlap=200):
    """Split text into chunks based on approximate token count."""
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        if end < len(text):
            break_point = text.rfind('\n\n', start, end)
            if break_point == -1:
                break_point = text.rfind('. ', start, end)
            if break_point != -1 and break_point > start:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap_chars if end < len(text) else end

    return chunks

def process_markdown_files(repo_path):
    """Process all markdown files in the repo and make chunks.json"""
    chunks_data = []
    md_files = list(Path(repo_path).rglob("*.md"))

    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                continue

            chunks = split_markdown_by_tokens(content)
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "id": f"{md_file.name}#chunk{i}",
                    "file": str(md_file.relative_to(repo_path)),
                    "content": chunk
                })
        except Exception as e:
            print(f"Error processing {md_file}: {e}")

    return chunks_data

if __name__ == "__main__":
    repo_path = "typescript-book/docs"  # path to your docs folder
    chunks = process_markdown_files(repo_path)

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed {len(chunks)} chunks")
