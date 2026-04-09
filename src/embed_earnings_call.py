import os

# 환경변수는 import 전에
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate용
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/processed/earnings_call_chunks_train.json")
INDEX_DIR = Path("vectordb/earnings_faiss_manual")
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.pkl"

MODEL_NAME = "intfloat/e5-base-v2"
BATCH_SIZE = 32


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    return text[:1500]   # 너무 긴 텍스트 방지


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # torch 쪽도 스레드 제한
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print(f"Loading chunks from: {INPUT_PATH}")
    rows = load_json(INPUT_PATH)
    print(f"Loaded {len(rows)} chunks")

    print("Loading SentenceTransformer on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 256

    texts = [f"passage: {clean_text(row['page_content'])}" for row in rows]

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
    index = faiss.IndexFlatIP(dim)
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