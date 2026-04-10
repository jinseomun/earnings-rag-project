from pathlib import Path
from typing import Any, Dict, List
import re

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
    Evidence 기반으로만 짧은 report를 생성한다.
    - claim을 그대로 반복하지 않음
    - evidence에 없는 내용은 쓰지 않음
    - evidence가 약하면 unsupported에 가깝게 보수적으로 작성
    """
    import re

    claim_text = get_claim_text(claim_item)
    evidence_text = evidence_item["text"]

    def normalize(text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def first_sentences(text: str, max_sentences: int = 2) -> str:
        text = normalize(text)
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return text[:250]
        return " ".join(parts[:max_sentences])

    def contains_any(text: str, keywords: list[str]) -> bool:
        lower = text.lower()
        return any(k.lower() in lower for k in keywords)

    evidence_norm = normalize(evidence_text)
    claim_norm = normalize(claim_text)

    # 1) product summary / units table 류
    if contains_any(evidence_norm, ["product summary", "iphone", "ipad", "mac", "units", "services"]):
        lines = [ln.strip() for ln in evidence_text.splitlines() if ln.strip()]
        compact = " ".join(lines)
        compact = re.sub(r"\s+", " ", compact)

        # iPhone 관련 claim이면 iPhone 주변만 최대한 짧게 잘라서 사용
        if "iphone" in claim_norm.lower() and "iphone" in compact.lower():
            m = re.search(
                r"(iphone.*?)(ipad|mac|services|other products|total apple|$)",
                compact,
                flags=re.IGNORECASE,
            )
            if m:
                snippet = m.group(1).strip()
                snippet = re.sub(r"\s+", " ", snippet)
                return f"Evidence-based finding: {snippet}"

        return f"Evidence-based finding: {compact[:320]}"

    # 2) press release headline / narrative 류
    if contains_any(evidence_norm, ["apple reports", "quarterly revenue", "net income", "fiscal", "quarter ended"]):
        summary = first_sentences(evidence_norm, max_sentences=2)
        return f"Evidence-based finding: {summary}"

    # 3) claim에 constant currency가 있는데 evidence에 그 표현이 없으면 보수적으로 처리
    if "constant currency" in claim_norm.lower() and "constant currency" not in evidence_norm.lower():
        return (
            "Evidence-based finding: The retrieved evidence does not clearly mention "
            "constant-currency revenue change, so this claim is not directly verified by the selected evidence."
        )

    # 4) evidence가 너무 숫자표/파편적 텍스트일 때 보수적으로 처리
    digit_count = sum(ch.isdigit() for ch in evidence_norm)
    alpha_count = sum(ch.isalpha() for ch in evidence_norm)

    if digit_count > alpha_count:
        return (
            "Evidence-based finding: The selected evidence is mostly tabular or numeric data, "
            "so the claim is only partially verifiable from this excerpt."
        )

    # 5) 기본 fallback: evidence 앞부분을 claim 반복 없이 요약
    fallback = first_sentences(evidence_norm, max_sentences=2)
    if len(fallback) < 40:
        fallback = evidence_norm[:220]

    return f"Evidence-based finding: {fallback}"


def clean_preview(text: str, max_len: int = 260) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len].rstrip() + " ..."
    return text


def clean_multiline_preview(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_len:
        return text[:max_len].rstrip() + " ..."
    return text


def format_score(score: Any) -> str:
    if score is None:
        return "N/A"
    try:
        return f"{float(score):.4f}"
    except Exception:
        return str(score)


def print_section_title(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_subsection_title(title: str):
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)


def print_key_value(label: str, value: Any):
    print(f"{label:<16} {value}")


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

        print_section_title(f"[Retrieval Eval] {idx}/{len(claims)} | claim_id={claim_id}")
        print_key_value("CLAIM:", clean_multiline_preview(claim_text, max_len=300))

        candidates = retrieve_top_k_for_claim(claim_item, top_k=top_k)

        if not candidates:
            print("\nNo candidates retrieved.")

        judged_candidates = []
        judge_labels = []

        for rank, cand in enumerate(candidates, start=1):
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

            print_subsection_title(f"Candidate {rank}")
            print_key_value("chunk_id:", cand.get("chunk_id"))
            print_key_value("score:", format_score(cand.get("score")))
            print_key_value("source_file:", cand.get("source_file") or "N/A")
            print_key_value("precheck:", precheck.get("rule_label"))
            print_key_value("judge:", label)
            print_key_value("reason:", clean_multiline_preview(judge_result.get("reason", ""), max_len=220))
            print_key_value("evidence:", clean_multiline_preview(evidence_text, max_len=380))

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
            print_section_title(f"[Report Eval] {idx}/{len(retrieval_records)} | claim_id={claim_id}")
            print("No best candidate found.")
            continue

        evidence_text = best_candidate["evidence_text"]
        generated_report = generate_report_for_claim(
            claim_item=claim_item,
            evidence_item={"text": evidence_text}
        )

        print_section_title(f"[Report Eval] {idx}/{len(retrieval_records)} | claim_id={claim_id}")
        print_key_value("CLAIM:", clean_multiline_preview(claim_text, max_len=300))
        print_key_value("LABEL:", best_candidate["judge_result"]["label"])
        print_key_value("SCORE:", format_score(best_candidate.get("score")))
        print_key_value("EVIDENCE:", clean_multiline_preview(evidence_text, max_len=380))
        print_key_value("REPORT:", clean_multiline_preview(generated_report, max_len=380))

        report_judge = judge_report_with_openai(
            client=client,
            model=report_judge_model,
            claim=claim_text,
            evidence=evidence_text,
            report=generated_report,
        )

        print_key_value("FAITHFULNESS:", report_judge.get("faithfulness"))
        print_key_value("HALLUCINATION:", report_judge.get("hallucination"))
        print_key_value("RELEVANCE:", report_judge.get("relevance_score"))
        print_key_value("REASON:", clean_multiline_preview(report_judge.get("reason", ""), max_len=220))

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


def print_summary(title: str, summary: Dict[str, Any]):
    print_section_title(title)
    for k, v in summary.items():
        if isinstance(v, float):
            print_key_value(f"{k}:", f"{v:.4f}")
        else:
            print_key_value(f"{k}:", v)


def main():
    client = OpenAI()

    print_section_title("Evaluation Config")
    print_key_value("CLAIMS_FILE:", CLAIMS_FILE.resolve())
    print_key_value("EMBED_FILE:", EMBED_FILE.resolve())

    claims = load_json(CLAIMS_FILE)

    print_subsection_title("Loaded claims sample")
    for c in claims[:5]:
        print(" -", clean_preview(get_claim_text(c), max_len=120))

    #claims = claims[:3]   # 디버깅용

    # 숫자 claim만 먼저 보고 싶으면 이거 켜기
    claims = [c for c in claims if any(ch.isdigit() for ch in get_claim_text(c))]

    print_subsection_title("Claims used for evaluation")
    for c in claims:
        print(" -", clean_preview(get_claim_text(c), max_len=160))

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

    print_summary("Retrieval Evaluation Summary", retrieval_eval_result["summary"])

    report_eval_result = evaluate_reports(
        retrieval_eval_result=retrieval_eval_result,
        claims_by_id=claims_by_id,
        client=client,
        report_judge_model="gpt-4.1-mini",
    )
    save_json(report_eval_result, REPORT_OUTPUT_FILE)

    print_summary("Report Evaluation Summary", report_eval_result["summary"])


if __name__ == "__main__":
    main()