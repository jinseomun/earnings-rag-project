import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "intfloat/e5-large-v2"

EMBED_FILE = Path("data/processed/sec_embeddings.json")
CLAIM_FILE = Path("outputs/claims.json")

model = SentenceTransformer(MODEL_NAME)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    with open(EMBED_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(CLAIM_FILE, "r", encoding="utf-8") as f:
        claims = json.load(f)

    query = "query: " + claims[0]["claim"]
    query_emb = model.encode([query])[0]

    scores = []
    for c in chunks:
        sim = cosine_similarity(query_emb, np.array(c["embedding"]))
        scores.append((sim, c["text"]))

    scores.sort(reverse=True)

    print("\nTop 3 retrieved chunks:\n")
    for s, text in scores[:3]:
        print(f"Score: {s:.4f}")
        print(text[:200])
        print("-" * 50)


if __name__ == "__main__":
    main()