# src/embed_sec.py
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/e5-large-v2"

INPUT_FILE = Path("data/processed/sec_chunks.json")
OUTPUT_FILE = Path("data/processed/sec_embeddings.json")

model = SentenceTransformer(MODEL_NAME, device="cpu")


def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found in input file.")
        return

    texts = ["passage: " + chunk["text"] for chunk in chunks]

    print(f"[INFO] Embedding {len(texts)} SEC chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,        #배치 사이즈 조절해서 안 터지게 함
        show_progress_bar=True
    )

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()