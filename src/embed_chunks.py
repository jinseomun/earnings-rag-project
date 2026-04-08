from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

MODEL_NAME = "intfloat/e5-large-v2"

INPUT_FILE = Path("data/processed/sec_chunks.json")
OUTPUT_FILE = Path("data/processed/sec_embeddings.json")

model = SentenceTransformer(MODEL_NAME)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = ["passage: " + c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print("Embedding done!")


if __name__ == "__main__":
    main()