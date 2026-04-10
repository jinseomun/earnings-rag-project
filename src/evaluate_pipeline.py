# src/evaluate_pipeline.py
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from eval_utils import (
    load_json,
    save_json,
    rule_precheck,
    summarize_retrieval_eval,
    summarize_report_eval,
)
from judge_evidence import judge_evidence_with_openai
from judge_report import judge_report_with_openai

from retrieval.retrieve import retrieve_top_chunks_for_claim

CLAIMS_FILE = Path("outputs/claims.json")
RETRIEVAL_OUTPUT_FILE = Path("outputs/retrieval_eval_results.json")
REPORT_OUTPUT_FILE = Path("outputs/report_eval_results.json")
EMBED_FILE = Path("data/processed/sec_embeddings.json")

def get_claim_text(claim_item: Dict[str, Any]) -> str:
    return claim_item.get("claim") or claim_item.get("text", "")


def retrieve_top_k_for_claim(claim_item, top_k: int = 5):
    claim_text = get_claim_text(claim_item)

    results = retrieve_top_chunks_for_claim(
        claim_text=claim_text,
        retrieval_top_n=20,
        final_top_k=top_k,
        embed_file=EMBED_FILE,
    )

    formatted = []
    for r in results:
        formatted.append({
            "chunk_id": r.get("chunk_id"),
            "text": r.get("text"),
            "score": r.get("rerank_score", r.get("retrieval_score", 0.0)),
            "source_file": r.get("metadata", {}).get("source_file"),
        })

    return formatted


def generate_report_for_claim(claim_item: Dict[str, Any], evidence_item: Dict[str, Any]) -> str:
    """
    TODO:
    여기에 네 report generation 코드를 연결해.
    지금은 예시 placeholder.
    """
    claim_text = get_claim_text(claim_item)
    evidence_text = evidence_item["text"]

    # 일단 placeholder
    return f"Claim: {claim_text}\nEvidence summary: {evidence_text[:300]}"


def evaluate_retrieval(
    claims: List[Dict[str, Any]],
    client: OpenAI,
    evidence_judge_model: str = "gpt-4.1-mini",
    top_k: int = 5,
) -> Dict[str, Any]:
    records = []

    for idx, claim_item in enumerate(claims, start=1):
        claim_id = claim_item.get("claim_id", idx)
        claim_text = get_claim_text(claim_item)

        print(f"[Retrieval Eval] {idx}/{len(claims)} claim_id={claim_id}")

        candidates = retrieve_top_k_for_claim(claim_item, top_k=top_k)

        judged_candidates = []
        judge_labels = []

        for cand in candidates:
            evidence_text = cand["text"]

            precheck = rule_precheck(claim_text, evidence_text)

            judge_result = judge_evidence_with_openai(
                client=client,
                model=evidence_judge_model,
                claim=claim_text,
                evidence=evidence_text,
            )

            label = judge_result["label"]
            judge_labels.append(label)

            judged_candidates.append({
                "chunk_id": cand.get("chunk_id"),
                "score": cand.get("score"),
                "source_file": cand.get("source_file"),
                "evidence_text": evidence_text,
                "rule_precheck": precheck,
                "judge_result": judge_result,
            })

        records.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "judge_labels": judge_labels,
            "candidates": judged_candidates,
        })

    summary = summarize_retrieval_eval(records)
    return {
        "summary": summary,
        "records": records,
    }


def select_best_supported_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    우선순위:
    supported > partially_supported > not_supported
    같은 label이면 score 높은 것
    """
    if not candidates:
        return None

    label_rank = {
        "supported": 2,
        "partially_supported": 1,
        "not_supported": 0,
    }

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            label_rank.get(x["judge_result"]["label"], -1),
            x.get("score", 0.0),
        ),
        reverse=True,
    )
    return sorted_candidates[0]


def evaluate_reports(
    retrieval_eval_result: Dict[str, Any],
    claims_by_id: Dict[Any, Dict[str, Any]],
    client: OpenAI,
    report_judge_model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    records = []

    retrieval_records = retrieval_eval_result["records"]

    for idx, rec in enumerate(retrieval_records, start=1):
        claim_id = rec["claim_id"]
        claim_item = claims_by_id[claim_id]
        claim_text = rec["claim_text"]

        best_candidate = select_best_supported_candidate(rec["candidates"])
        if best_candidate is None:
            continue

        evidence_text = best_candidate["evidence_text"]
        generated_report = generate_report_for_claim(
            claim_item=claim_item,
            evidence_item={"text": evidence_text}
        )

        print(f"[Report Eval] {idx}/{len(retrieval_records)} claim_id={claim_id}")

        report_judge = judge_report_with_openai(
            client=client,
            model=report_judge_model,
            claim=claim_text,
            evidence=evidence_text,
            report=generated_report,
        )

        records.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "selected_evidence": evidence_text,
            "generated_report": generated_report,
            "report_judge": report_judge,
        })

    summary = summarize_report_eval(records)
    return {
        "summary": summary,
        "records": records,
    }


def main():
    client = OpenAI()

    claims = load_json(CLAIMS_FILE)

    # 숫자 claim만 먼저 보고 싶으면 이거 켜기
    claims = [c for c in claims if any(ch.isdigit() for ch in get_claim_text(c))]

    claims_by_id = {}
    for i, c in enumerate(claims, start=1):
        claim_id = c.get("claim_id", i)
        c["claim_id"] = claim_id
        claims_by_id[claim_id] = c

    retrieval_eval_result = evaluate_retrieval(
        claims=claims,
        client=client,
        evidence_judge_model="gpt-4.1-mini",
        top_k=5,
    )
    save_json(retrieval_eval_result, RETRIEVAL_OUTPUT_FILE)

    print("\n=== Retrieval Evaluation Summary ===")
    for k, v in retrieval_eval_result["summary"].items():
        print(f"{k}: {v}")

    report_eval_result = evaluate_reports(
        retrieval_eval_result=retrieval_eval_result,
        claims_by_id=claims_by_id,
        client=client,
        report_judge_model="gpt-4.1-mini",
    )
    save_json(report_eval_result, REPORT_OUTPUT_FILE)

    print("\n=== Report Evaluation Summary ===")
    for k, v in report_eval_result["summary"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()