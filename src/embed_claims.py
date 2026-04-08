from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

MODEL_NAME = "intfloat/e5-large-v2"

INPUT_FILE = Path("outputs/claims.json")
OUTPUT_FILE = Path("data/processed/claim_embeddings.json")

model = SentenceTransformer(MODEL_NAME)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        claims = json.load(f)

    texts = ["passage: " + c["claim"] for c in claims]

    embeddings = model.encode(texts, show_progress_bar=True)

    for i, emb in enumerate(embeddings):
        claims[i]["embedding"] = emb.tolist()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(claims, f, ensure_ascii=False, indent=2)

    print("Embedding done!")


if __name__ == "__main__":
    main()