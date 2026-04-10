from pathlib import Path

from claims.claim_extraction import extract_claims_with_llama
from retrieval.retrieve import retrieve_top_chunks_for_claim
from parsing.query_parser import parse_user_query
from parsing.event_resolver import resolve_transcript
from reports.report_generator import generate_report
from sec.sec_fetcher import fetch_sec_filings_for_ticker_near_date
from claims.transcript_loader import load_transcript_from_json

# 기존 chunk / embedding 생성 함수에 맞게 import 이름 바꾸기
from sec.chunk_sec import build_sec_chunks_from_txt_files
from sec.embed_sec import build_sec_embeddings_from_chunks


SEC_TEXT_DIR = Path("data/raw/sec/text")
SEC_CHUNKS_PATH = Path("data/processed/sec_chunks.json")
SEC_EMBEDDINGS_PATH = Path("data/processed/sec_embeddings.json")

DEBUG = True

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


def extract_filing_date_from_path(path: Path):
    """
    예:
    aapl_2016-07-27_10-Q_clean.txt
    """
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    return parts[1]


def ensure_sec_filings(ticker: str, target_date: str, limit: int = 3):
    """
    transcript 날짜(target_date) 이전/당일 SEC txt 우선 재사용.
    없으면 SEC에서 target_date 근처 filings를 fetch.
    """
    existing_files = sorted(SEC_TEXT_DIR.glob(f"{ticker.lower()}_*.txt"))

    matched = []
    for path in existing_files:
        filing_date = extract_filing_date_from_path(path)
        if filing_date is None:
            continue

        if filing_date <= target_date:
            matched.append(path)

    matched = sorted(
        matched,
        key=lambda p: extract_filing_date_from_path(p),
        reverse=True,
    )[:limit]

    if matched:
        print(
            f"[INFO] Reusing SEC txt files near {target_date} for {ticker}: {len(matched)} found"
        )
        for p in matched:
            print("   ", p.name)
        return matched

    print(f"[INFO] No local SEC txt files found near {target_date} for {ticker}. Fetching from SEC...")

    results = fetch_sec_filings_for_ticker_near_date(
        ticker=ticker,
        target_date=target_date,
        form_types=["10-Q", "10-K", "8-K"],
        limit=limit,
    )

    return [Path(item["clean_text_path"]) for item in results]


def rebuild_sec_knowledge_base(sec_txt_files):
    """
    선택된 SEC txt들로 sec_chunks.json, sec_embeddings.json 다시 생성
    """
    print("[INFO] Rebuilding SEC chunks...")
    build_sec_chunks_from_txt_files(
        txt_paths=sec_txt_files,
        output_path=SEC_CHUNKS_PATH,
    )

    print("[INFO] Rebuilding SEC embeddings...")
    build_sec_embeddings_from_chunks(
        chunks_path=SEC_CHUNKS_PATH,
        output_path=SEC_EMBEDDINGS_PATH,
    )

    print("[INFO] SEC knowledge base ready.")
    print(f"       chunks: {SEC_CHUNKS_PATH}")
    print(f"       embeddings: {SEC_EMBEDDINGS_PATH}")


def main():
    user_query = input("질문을 입력하세요: ").strip()

    if not user_query:
        print("질문이 비어 있습니다.")
        return

    query_info = parse_user_query(user_query)
    if DEBUG:
        print("query_info:", query_info)

    if not query_info["company"]:
        print("회사명을 인식하지 못했습니다.")
        return

    event_info = resolve_transcript(query_info)
    if not event_info:
        print("해당 조건에 맞는 transcript를 찾을 수 없습니다.")
        return

    if DEBUG:
        print("event_info:", event_info)

    company = event_info["company"]
    ticker = event_info.get("ticker") or TICKER_MAP.get(company) or TICKER_MAP.get(
        company.upper()
    )
    target_date = event_info.get("date")

    sec_files = []
    if ticker and target_date:
        sec_files = ensure_sec_filings(
            ticker=ticker,
            target_date=target_date,
            limit=3,
        )

        if sec_files:
            print(f"[INFO] SEC txt ready: {len(sec_files)} files")
            rebuild_sec_knowledge_base(sec_files)
        else:
            print("[WARN] No usable SEC txt files found. Continue with transcript only.")
    else:
        print(f"[WARN] Missing ticker/date for {company}. Continue with transcript only.")

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
        retrieved = retrieve_top_chunks_for_claim(
            claim["claim"],
            final_top_k=2,
            embed_file=SEC_EMBEDDINGS_PATH,   # 중요: 방금 재생성한 임베딩 파일 사용
        )

        if DEBUG:
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