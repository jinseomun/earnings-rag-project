import json
import re
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def build_retrieval_query(claim_text: str) -> str:
    """
    claim 원문을 retrieval-friendly query로 확장
    - SEC 문서 표현(net sales, gross margin percentage 등) 반영
    - 비교 표현(yoy, sequential, basis points) 확장
    """
    q = normalize_text(claim_text)

    replacements = [
        ("year-over-year", "year over year yoy"),
        ("yoy", "year over year yoy"),
        ("sequentially", "sequential quarter over quarter qoq"),
        ("quarter-over-quarter", "quarter over quarter qoq sequential"),
        ("qoq", "quarter over quarter qoq sequential"),
        ("basis points", "basis points bps"),
        ("bps", "basis points bps"),
        ("revenue", "revenue net sales"),
        ("gross margin", "gross margin gross margin percentage"),
        ("iphone revenue", "iphone revenue iphone net sales"),
        ("services revenue", "services revenue services net sales"),
        ("products gross margin", "products gross margin products margin"),
    ]

    for src, tgt in replacements:
        q = q.replace(src, tgt)

    # metric 힌트 강화
    hints = []

    if "iphone" in q:
        hints.extend(["iphone", "net sales", "product performance"])
    if "services" in q:
        hints.extend(["services", "net sales", "dollars in millions"])
    if "gross margin" in q:
        hints.extend(["gross margin percentage", "cost of sales"])
    if "revenue" in q or "net sales" in q:
        hints.extend(["net sales", "change", "dollars in millions"])
    if "year over year" in q or "yoy" in q:
        hints.extend(["change", "prior year", "current year"])
    if "sequential" in q or "quarter over quarter" in q:
        hints.extend(["previous quarter", "current quarter", "quarter"])

    if hints:
        q = q + " " + " ".join(hints)

    return "query: " + q


def build_rerank_query(claim_text: str) -> str:
    """
    reranker에는 '비슷한 문장'보다 '검증 가능한 evidence'를 찾도록 약간 유도
    """
    return (
        "Find financial evidence that can verify this claim using relevant metric, "
        "period, comparison direction, and raw numbers if available: "
        + claim_text
    )


def extract_numbers(text: str):
    pattern = r'\$?\d+(?:,\d+)?(?:\.\d+)?%?'
    return re.findall(pattern, text)


def numeric_density_bonus(text: str) -> float:
    """
    숫자/표 중심 chunk에 약한 보너스
    너무 세면 숫자만 많은 엉뚱한 표가 올라오니 작게만 반영
    """
    nums = extract_numbers(text)
    return min(len(nums) * 0.005, 0.08)


def metric_bonus(claim_text: str, evidence_text: str) -> float:
    """
    claim의 핵심 metric이 evidence에도 있으면 약한 보너스
    """
    claim = normalize_text(claim_text)
    evidence = normalize_text(evidence_text)

    bonus = 0.0

    metric_groups = [
        (["revenue", "net sales"], ["revenue", "net sales"]),
        (["iphone"], ["iphone"]),
        (["services"], ["services"]),
        (["gross margin"], ["gross margin"]),
        (["products"], ["products"]),
        (["mac"], ["mac"]),
        (["ipad"], ["ipad"]),
    ]

    for claim_terms, evidence_terms in metric_groups:
        if any(t in claim for t in claim_terms) and any(t in evidence for t in evidence_terms):
            bonus += 0.03

    return min(bonus, 0.12)


def comparison_bonus(claim_text: str, evidence_text: str) -> float:
    """
    YoY / sequential / basis points 같은 비교 표현이 evidence에도 있으면 약한 보너스
    """
    claim = normalize_text(claim_text)
    evidence = normalize_text(evidence_text)

    bonus = 0.0

    yoy_terms = ["year over year", "yoy"]
    seq_terms = ["sequential", "quarter over quarter", "qoq"]
    bps_terms = ["basis points", "bps"]
    change_terms = ["change", "%"]

    if any(t in claim for t in yoy_terms):
        if any(t in evidence for t in yoy_terms + change_terms):
            bonus += 0.02

    if any(t in claim for t in seq_terms):
        if any(t in evidence for t in seq_terms + change_terms):
            bonus += 0.02

    if any(t in claim for t in bps_terms):
        if any(t in evidence for t in bps_terms + change_terms):
            bonus += 0.02

    return min(bonus, 0.04)


def compute_retrieval_score(claim_text: str, chunk_text: str, query_emb: np.ndarray, chunk_emb: np.ndarray) -> float:
    base_sim = cosine_similarity(query_emb, chunk_emb)
    bonus = (
        numeric_density_bonus(chunk_text)
        + metric_bonus(claim_text, chunk_text)
        + comparison_bonus(claim_text, chunk_text)
    )
    return float(base_sim + bonus)


def retrieve_candidates_for_claim(claim_text: str, embed_file: Path, top_n: int = 20):
    chunks = load_chunks(embed_file)

    query = build_retrieval_query(claim_text)
    query_emb = embed_model.encode([query], convert_to_numpy=True)[0]

    scored = []
    for c in chunks:
        chunk_emb = np.array(c["embedding"], dtype=np.float32)

        text = c.get("page_content") or c.get("text", "")
        metadata = c.get("metadata", {})

        sim = compute_retrieval_score(
            claim_text=claim_text,
            chunk_text=text,
            query_emb=query_emb,
            chunk_emb=chunk_emb,
        )

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

    rerank_query = build_rerank_query(claim_text)
    pairs = [[rerank_query, c["text"]] for c in candidates]
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