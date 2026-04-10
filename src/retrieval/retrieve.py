import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL_NAME = "intfloat/e5-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_EMBED_FILE = Path("data/processed/sec_embeddings.json")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
rerank_model = CrossEncoder(RERANK_MODEL_NAME)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def load_chunks(embed_file: Path):
    with open(embed_file, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_candidates_for_claim(claim_text: str, embed_file: Path, top_n: int = 20):
    chunks = load_chunks(embed_file)

    query = "query: " + claim_text
    query_emb = embed_model.encode([query], convert_to_numpy=True)[0]

    scored = []
    for c in chunks:
        chunk_emb = np.array(c["embedding"], dtype=np.float32)

        text = c.get("page_content") or c.get("text", "")
        metadata = c.get("metadata", {})

        sim = cosine_similarity(query_emb, chunk_emb)

        scored.append({
            "retrieval_score": sim,
            "text": text,
            "chunk_id": metadata.get("chunk_id", c.get("chunk_id")),
            "metadata": metadata,
        })

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:top_n]


def rerank_chunks(claim_text: str, candidates: list, top_k: int = 3):
    if not candidates:
        return []

    pairs = [[claim_text, c["text"]] for c in candidates]
    rerank_scores = rerank_model.predict(pairs)

    reranked = []
    for c, score in zip(candidates, rerank_scores):
        item = c.copy()
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def retrieve_top_chunks_for_claim(
    claim_text: str,
    retrieval_top_n: int = 20,
    final_top_k: int = 3,
    embed_file: Path = DEFAULT_EMBED_FILE,
):
    candidates = retrieve_candidates_for_claim(
        claim_text=claim_text,
        embed_file=embed_file,
        top_n=retrieval_top_n,
    )

    final_results = rerank_chunks(
        claim_text=claim_text,
        candidates=candidates,
        top_k=final_top_k,
    )

    return final_results