# src/pipeline.py
from pathlib import Path

from claim_extraction import extract_claims_with_llama
from retrieve import retrieve_top_chunks_for_claim
from query_parser import parse_user_query
from event_resolver import resolve_transcript
from report_generator import generate_report
from sec_fetcher import fetch_recent_sec_filings_for_ticker
from transcript_loader import load_transcript_from_json

SEC_TEXT_DIR = Path("data/raw/sec/text")

TICKER_MAP = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "fb": "META",
    "meta platforms": "META",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "goog": "GOOGL",
    "googl": "GOOGL",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "intc": "INTC",
    "intel": "INTC",
    "mu": "MU",
    "micron": "MU",
    "csco": "CSCO",
    "cisco": "CSCO",
    "asml": "ASML",
}


def select_relevant_claims(claims, user_query, top_n=3):
    query_words = set(user_query.lower().split())

    scored = []
    for claim in claims:
        claim_text = claim["claim"].lower()
        overlap = sum(1 for w in query_words if w in claim_text)
        scored.append((overlap, claim))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [c for score, c in scored[:top_n] if score > 0]

    if not selected:
        selected = claims[:top_n]

    return selected


def ensure_sec_filings(company: str, ticker: str, limit: int = 3):
    existing_files = list(SEC_TEXT_DIR.glob(f"{ticker.lower()}_*.txt"))

    if existing_files:
        print(f"[INFO] Reusing existing SEC txt files for {ticker}: {len(existing_files)} found")
        return existing_files

    print(f"[INFO] No local SEC txt files found for {ticker}. Fetching from SEC...")
    results = fetch_recent_sec_filings_for_ticker(
        ticker=ticker,
        form_types=["10-Q", "10-K", "8-K"],
        limit=limit,
    )

    return [Path(item["clean_text_path"]) for item in results]


def main():
    user_query = input("질문을 입력하세요: ").strip()

    if not user_query:
        print("질문이 비어 있습니다.")
        return

    query_info = parse_user_query(user_query)
    print("query_info:", query_info)

    if not query_info["company"]:
        print("회사명을 인식하지 못했습니다.")
        return

    event_info = resolve_transcript(query_info)
    if not event_info:
        print("해당 조건에 맞는 transcript를 찾을 수 없습니다.")
        return

    company = event_info["company"]
    ticker = event_info.get("ticker") or TICKER_MAP.get(company) or TICKER_MAP.get(
        company.upper()
    )

    sec_files = []
    if ticker:
        sec_files = ensure_sec_filings(company, ticker, limit=3)
        if sec_files:
            print(f"[INFO] SEC txt ready: {len(sec_files)} files")
        else:
            print("[WARN] No usable SEC txt files found. Continue with transcript only.")
    else:
        print(f"[WARN] No ticker mapping found for {company}. Continue with transcript only.")

    print(
        f"[INFO] Selected transcript: call_id={event_info.get('call_id')} date={event_info.get('date')}"
    )

    transcript_text = load_transcript_from_json(event_info["call_id"])
    claims = extract_claims_with_llama(transcript_text)

    if not claims:
        print("No claims extracted.")
        return

    relevant_claims = select_relevant_claims(claims, user_query, top_n=3)

    analyzed_claims = []
    for claim in relevant_claims:
        retrieved = retrieve_top_chunks_for_claim(claim["claim"], final_top_k=2)

        print("\n[DEBUG] CLAIM:", claim["claim"])
        for i, r in enumerate(retrieved, start=1):
            print(f"  [Evidence {i}]")
            print("   chunk_id:", r.get("chunk_id"))
            print("   retrieval_score:", r.get("retrieval_score"))
            print("   rerank_score:", r.get("rerank_score"))
            print("   text:", r.get("text", "")[:200])
            print("-" * 60)

        analyzed_claims.append(
            {
                "claim": claim,
                "evidence": retrieved,
            }
        )

    report = generate_report(user_query, event_info, analyzed_claims)
    print("\n" + report)


if __name__ == "__main__":
    main()
