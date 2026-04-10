import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1차 retrieval용 임베딩 모델
EMBED_MODEL_NAME = "intfloat/e5-base-v2"

# 2차 reranking용 모델
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBED_FILE = Path("data/processed/sec_embeddings.json")

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


def retrieve_candidates_for_claim(claim_text: str, top_n: int = 20):
    """
    1차 retrieval:
    claim을 query로 임베딩하고,
    sec chunk embedding과 cosine similarity를 계산해
    상위 top_n개 후보를 반환
    """
    chunks = load_chunks(EMBED_FILE)

    query = "query: " + claim_text
    query_emb = embed_model.encode([query], convert_to_numpy=True)[0]

    scored = []
    for c in chunks:
        chunk_emb = np.array(c["embedding"], dtype=np.float32)
        sim = cosine_similarity(query_emb, chunk_emb)

        scored.append({
            "retrieval_score": sim,
            "text": c["text"],
            "chunk_id": c.get("chunk_id")
        })

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:top_n]


def rerank_chunks(claim_text: str, candidates: list, top_k: int = 3):
    """
    2차 reranking:
    retrieval로 뽑은 후보들에 대해
    [claim, chunk_text] 쌍을 CrossEncoder에 넣어 재점수화
    """
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
    final_top_k: int = 3
):
    """
    전체 retrieval + reranking 파이프라인
    """
    candidates = retrieve_candidates_for_claim(
        claim_text=claim_text,
        top_n=retrieval_top_n
    )

    final_results = rerank_chunks(
        claim_text=claim_text,
        candidates=candidates,
        top_k=final_top_k
    )

    return final_results


def main():
    claim = input("Claim: ").strip()

    if not claim:
        print("빈 claim은 검색할 수 없습니다.")
        return

    results = retrieve_top_chunks_for_claim(
        claim_text=claim,
        retrieval_top_n=20,
        final_top_k=3
    )

    print("\nTop reranked chunks:\n")

    if not results:
        print("검색 결과가 없습니다.")
        return

    for i, r in enumerate(results, start=1):
        print(f"[Final Rank {i}]")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Retrieval score: {r['retrieval_score']:.8f}")
        print(f"Rerank score: {r['rerank_score']:.8f}")
        print(r["text"][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()