import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/processed/earnings_call_chunks_train.json")
INDEX_DIR = Path("vectordb/earnings_faiss_manual")
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.pkl"

MODEL_NAME = "intfloat/e5-base-v2"
BATCH_SIZE = 16  # CPU면 8~32 정도 추천


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from: {INPUT_PATH}")
    rows = load_json(INPUT_PATH)
    print(f"Loaded {len(rows)} chunks")

    print("Loading SentenceTransformer on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = [f"passage: {row['page_content'].strip()}" for row in rows]

    all_embeddings = []
    for i, batch_texts in enumerate(batch_iter(texts, BATCH_SIZE)):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(texts))
        print(f"Encoding batch {start} ~ {end}")

        emb = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print("Embeddings shape:", embeddings.shape)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity용
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index to: {INDEX_PATH}")

    metadata_store = []
    for row in rows:
        metadata_store.append({
            "page_content": row["page_content"],
            "metadata": row["metadata"]
        })

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    main()