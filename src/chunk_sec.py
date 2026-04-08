from pathlib import Path
import json

INPUT_DIR = Path("data/raw/sec")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 300  # 대충 300~500 토큰 느낌


def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def main():
    file_path = INPUT_DIR / "apple_10q.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "chunk_id": i,
            "text": chunk
        })

    with open(OUTPUT_DIR / "sec_chunks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} chunks")


if __name__ == "__main__":
    main()