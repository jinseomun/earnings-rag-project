# src/sec_fetcher.py
import json
import time
from pathlib import Path
from typing import Optional, List, Dict

import requests

BASE_DATA_URL = "https://data.sec.gov"
BASE_ARCHIVES_URL = "https://www.sec.gov/Archives"

RAW_SEC_DIR = Path("data/raw/sec")
RAW_SEC_DIR.mkdir(parents=True, exist_ok=True)

# SEC는 user-agent 명시를 권장/요구
HEADERS = {
    "User-Agent": "earnings-rag-project munjinseo@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


def zero_pad_cik(cik: str | int) -> str:
    return str(cik).zfill(10)


def get_company_tickers() -> list[dict]:
    """
    SEC 제공 ticker/CIK/company name 매핑 파일 다운로드
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # SEC json은 {"0": {...}, "1": {...}} 형태
    return list(data.values())


def find_cik_by_ticker(ticker: str) -> Optional[str]:
    ticker = ticker.upper()
    companies = get_company_tickers()

    for item in companies:
        if item["ticker"].upper() == ticker:
            return zero_pad_cik(item["cik_str"])

    return None


def get_submissions_json(cik: str) -> dict:
    cik = zero_pad_cik(cik)
    url = f"{BASE_DATA_URL}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_recent_filings(submissions_json: dict) -> List[Dict]:
    """
    submissions JSON의 recent filings를 row 형태로 변환
    """
    recent = submissions_json.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_documents = recent.get("primaryDocument", [])
    primary_doc_descriptions = recent.get("primaryDocDescription", [])

    filings = []
    n = min(
        len(forms),
        len(accession_numbers),
        len(filing_dates),
        len(primary_documents),
        len(primary_doc_descriptions),
    )

    for i in range(n):
        filings.append(
            {
                "form": forms[i],
                "accession_number": accession_numbers[i],
                "filing_date": filing_dates[i],
                "primary_document": primary_documents[i],
                "primary_doc_description": primary_doc_descriptions[i],
            }
        )

    return filings


def filter_filings(
    filings: List[Dict],
    form_types: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict]:
    if form_types is None:
        form_types = ["10-Q", "10-K", "8-K"]

    results = [f for f in filings if f["form"] in form_types]
    return results[:limit]


def build_filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """
    filing 원문 문서 URL 생성
    예:
    https://www.sec.gov/Archives/edgar/data/{CIK no leading zeros}/{accession no dashes}/{primary_document}
    """
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return f"{BASE_ARCHIVES_URL}/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_document}"


def download_filing_text(url: str) -> str:
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def save_filing(company: str, filing: Dict, text: str) -> Path:
    filing_date = filing["filing_date"]
    form = filing["form"].replace("/", "-")
    filename = f"{company.lower()}_{filing_date}_{form}.txt"
    out_path = RAW_SEC_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return out_path


def fetch_recent_sec_filings_for_ticker(
    ticker: str,
    form_types: Optional[List[str]] = None,
    limit: int = 3,
    sleep_sec: float = 0.2,
):
    cik = find_cik_by_ticker(ticker)
    if not cik:
        raise ValueError(f"CIK not found for ticker: {ticker}")

    submissions = get_submissions_json(cik)
    company_name = submissions.get("name", ticker.upper())

    filings = get_recent_filings(submissions)
    selected = filter_filings(filings, form_types=form_types, limit=limit)

    saved_files = []

    for filing in selected:
        doc_url = build_filing_document_url(
            cik=cik,
            accession_number=filing["accession_number"],
            primary_document=filing["primary_document"],
        )
        print(f"[INFO] Downloading {filing['form']} {filing['filing_date']} -> {doc_url}")

        text = download_filing_text(doc_url)
        path = save_filing(ticker, filing, text)

        saved_files.append(
            {
                "company_name": company_name,
                "ticker": ticker.upper(),
                "cik": cik,
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "path": str(path),
                "url": doc_url,
            }
        )

        time.sleep(sleep_sec)

    return saved_files


def main():
    ticker = input("Ticker를 입력하세요 (예: AAPL): ").strip().upper()
    results = fetch_recent_sec_filings_for_ticker(
        ticker=ticker,
        form_types=["10-Q", "10-K", "8-K"],
        limit=3,
    )

    print("\n=== Saved Filings ===")
    for item in results:
        print(item)


if __name__ == "__main__":
    main()