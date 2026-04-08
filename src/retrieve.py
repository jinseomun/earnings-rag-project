import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "intfloat/e5-large-v2"
EMBED_FILE = Path("data/processed/sec_embeddings.json")

model = SentenceTransformer(MODEL_NAME)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_top_chunks_for_claim(claim_text, top_k=3):
    with open(EMBED_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    query = "query: " + claim_text
    query_emb = model.encode([query])[0]

    scored = []
    for c in chunks:
        sim = cosine_similarity(query_emb, np.array(c["embedding"]))
        scored.append({
            "score": float(sim),
            "text": c["text"],
            "chunk_id": c.get("chunk_id")
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def main():
    claim = input("Claim: ").strip()
    results = retrieve_top_chunks_for_claim(claim, top_k=3)

    print("\nTop retrieved chunks:\n")
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(r["text"][:300])
        print("-" * 50)


if __name__ == "__main__":
    main()