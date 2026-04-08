# src/sec_fetcher.py
import re
import time
from pathlib import Path
from typing import Optional, List, Dict
from bs4 import BeautifulSoup

import requests


BASE_DATA_URL = "https://data.sec.gov"
BASE_ARCHIVES_URL = "https://www.sec.gov/Archives"

RAW_SEC_TEXT_DIR = Path("data/raw/sec/text")
RAW_SEC_TEXT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS_DATA = {
    "User-Agent": "earnings-rag-project your_email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

HEADERS_WWW = {
    "User-Agent": "earnings-rag-project your_email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


def zero_pad_cik(cik: str | int) -> str:
    return str(cik).zfill(10)


def get_company_tickers() -> list[dict]:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS_WWW, timeout=30)
    resp.raise_for_status()
    data = resp.json()
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
    resp = requests.get(url, headers=HEADERS_DATA, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_recent_filings(submissions_json: dict) -> List[Dict]:
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


def build_filing_folder_url(cik: str, accession_number: str) -> str:
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return f"{BASE_ARCHIVES_URL}/edgar/data/{cik_no_zeros}/{accession_no_dashes}"


def build_filing_document_url(cik: str, accession_number: str, filename: str) -> str:
    folder_url = build_filing_folder_url(cik, accession_number)
    return f"{folder_url}/{filename}"


def fetch_archive_index_json(cik: str, accession_number: str) -> Optional[dict]:
    folder_url = build_filing_folder_url(cik, accession_number)
    index_url = f"{folder_url}/index.json"

    resp = requests.get(index_url, headers=HEADERS_WWW, timeout=30)
    if resp.status_code != 200:
        return None

    return resp.json()


def get_archive_items(index_json: dict) -> List[dict]:
    return index_json.get("directory", {}).get("item", [])


def is_probably_useful_txt(filename: str) -> bool:
    name = filename.lower()

    if not name.endswith(".txt"):
        return False

    bad_keywords = [
        ".xsd",
        ".xml",
        "_cal",
        "_def",
        "_lab",
        "_pre",
        "_sch",
        "schema",
        "def.xml",
        "lab.xml",
        "pre.xml",
        "cal.xml",
    ]
    if any(bad in name for bad in bad_keywords):
        return False

    return True


def score_txt_candidate(filename: str, primary_document: str) -> int:
    name = filename.lower()
    primary = primary_document.lower()

    score = 0

    if name == primary and name.endswith(".txt"):
        score += 100

    primary_base = primary.rsplit(".", 1)[0]
    name_base = name.rsplit(".", 1)[0]

    if name.endswith(".txt") and name_base == primary_base:
        score += 80

    if name.endswith(".txt"):
        score += 30

    if "complete" in name or "submission" in name:
        score -= 10

    if primary_base in name:
        score += 10

    return score


def choose_best_txt_filename(filing: Dict, archive_items: List[dict]) -> Optional[str]:
    primary_document = filing["primary_document"]

    txt_candidates = []
    for item in archive_items:
        name = item.get("name", "")
        if is_probably_useful_txt(name):
            txt_candidates.append(name)

    if not txt_candidates:
        return None

    ranked = sorted(
        txt_candidates,
        key=lambda x: score_txt_candidate(x, primary_document),
        reverse=True,
    )
    return ranked[0]


def download_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS_WWW, timeout=30)
    resp.raise_for_status()
    return resp.text


def clean_sec_filing_text(raw_text: str) -> str:
    """
    SEC raw submission txt -> 분석용 clean text
    """

    # 1) SEC HEADER 제거
    raw_text = re.sub(
        r"<SEC-HEADER>.*?</SEC-HEADER>",
        " ",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2) uuencoded graphic/image block 제거
    raw_text = re.sub(
        r"begin\s+\d{3}\s+.*?\n.*?\nend",
        " ",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 3) TEXT 블록만 추출
    text_match = re.search(
        r"<TEXT>(.*)</TEXT>",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if text_match:
        raw_text = text_match.group(1)

    # 4) XBRL wrapper 제거
    raw_text = re.sub(r"</?XBRL[^>]*>", " ", raw_text, flags=re.IGNORECASE)

    # 5) HTML/XBRL 파싱
    soup = BeautifulSoup(raw_text, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # 6) 남은 SGML/HTML 태그 제거
    text = re.sub(r"<[^>]+>", " ", text)

    # 7) 줄 단위 잡음 제거
    cleaned_lines = []
    noisy_patterns = [
        "xmlns:",
        "xbrli:",
        "xbrldi:",
        "us-gaap:",
        "dei:",
        "link:",
        "xlink:",
        "contextref",
        "unitref",
        "ix:nonnumeric",
        "ix:nonfraction",
        "aapl:",
    ]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        lower = line.lower()

        # 메타/헤더성 줄 제거
        if lower.startswith("accession number:"):
            continue
        if lower.startswith("conformed submission type:"):
            continue
        if lower.startswith("filed as of date:"):
            continue
        if lower.startswith("date as of change:"):
            continue
        if lower.startswith("public document count:"):
            continue
        if lower.startswith("central index key:"):
            continue
        if lower.startswith("standard industrial classification:"):
            continue

        # XBRL namespace/기술 문자열 위주 줄 제거
        if sum(p in lower for p in noisy_patterns) >= 2:
            continue

        # 너무 짧고 의미 없는 값들 제거
        if len(line) <= 3 and lower in {"true", "false", "—"}:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 8) Exhibit 구간 제거 시도
    # 8-K에서 EX-10.1 같은 첨부문서가 붙는 경우가 많음
    text = re.split(r"\nEX-\d+(\.\d+)?\n", text, maxsplit=1, flags=re.IGNORECASE)[0]

    # 9) 공백 정리
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()

def save_clean_text(company: str, filing: Dict, text: str, suffix_name: str) -> Path:
    filing_date = filing["filing_date"]
    form = filing["form"].replace("/", "-")
    filename = f"{company.lower()}_{filing_date}_{form}_{suffix_name}.txt"
    out_path = RAW_SEC_TEXT_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return out_path


def fetch_one_filing_txt_only(cik: str, ticker: str, filing: Dict) -> Optional[Dict]:
    accession_number = filing["accession_number"]

    index_json = fetch_archive_index_json(cik, accession_number)
    archive_items = get_archive_items(index_json) if index_json else []

    chosen_filename = choose_best_txt_filename(filing, archive_items)

    if not chosen_filename:
        print(
            f"[SKIP] {filing['form']} {filing['filing_date']} | "
            f"No usable .txt source found."
        )
        return None

    source_url = build_filing_document_url(cik, accession_number, chosen_filename)

    print(
        f"[INFO] {filing['form']} {filing['filing_date']} | "
        f"txt source found: {chosen_filename}"
    )

    raw_text = download_text(source_url)
    clean_text = clean_sec_filing_text(raw_text)
    text_path = save_clean_text(ticker, filing, clean_text, "clean")

    return {
        "ticker": ticker.upper(),
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "accession_number": accession_number,
        "source_filename": chosen_filename,
        "source_url": source_url,
        "clean_text_path": str(text_path),
    }


def fetch_recent_sec_filings_for_ticker(
    ticker: str,
    form_types: Optional[List[str]] = None,
    limit: int = 3,
    sleep_sec: float = 0.2,
) -> List[Dict]:
    cik = find_cik_by_ticker(ticker)
    if not cik:
        raise ValueError(f"CIK not found for ticker: {ticker}")

    submissions = get_submissions_json(cik)
    filings = get_recent_filings(submissions)
    selected = filter_filings(filings, form_types=form_types, limit=limit)

    results = []
    for filing in selected:
        item = fetch_one_filing_txt_only(cik, ticker, filing)
        if item is not None:
            results.append(item)
        time.sleep(sleep_sec)

    return results


def main():
    ticker = input("Ticker를 입력하세요 (예: AAPL): ").strip().upper()

    results = fetch_recent_sec_filings_for_ticker(
        ticker=ticker,
        form_types=["10-Q", "10-K", "8-K"],
        limit=3,
    )

    print("\n=== Saved TXT Filings ===")
    if not results:
        print("No usable txt filings found.")
        return

    for item in results:
        print(item)


if __name__ == "__main__":
    main()