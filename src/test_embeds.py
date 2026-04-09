import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/e5-base-v2"

INPUT_FILE = Path("data/processed/earnings_call_chunks_train.json")
OUTPUT_FILE = Path("data/processed/earnings_call_embeddings_train.json")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    return text[:1500]


def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 256

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found in input file.")
        return

    texts = [f"passage: {clean_text(chunk['page_content'])}" for chunk in chunks]

    print(f"[INFO] Embedding {len(texts)} earnings call chunks...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Saved embeddings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()