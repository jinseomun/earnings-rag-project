# src/sec_fetcher.py
import re
import time
from datetime import datetime
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


def fetch_json(url: str, use_data_headers: bool = True) -> dict:
    headers = HEADERS_DATA if use_data_headers else HEADERS_WWW
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_filings_columnar(block: dict) -> List[Dict]:
    forms = block.get("form", [])
    accession_numbers = block.get("accessionNumber", [])
    filing_dates = block.get("filingDate", [])
    primary_documents = block.get("primaryDocument", [])
    primary_doc_descriptions = block.get("primaryDocDescription", [])

    n = min(
        len(forms),
        len(accession_numbers),
        len(filing_dates),
        len(primary_documents),
        len(primary_doc_descriptions),
    )

    filings = []
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


def get_all_filings(submissions_json: dict) -> List[Dict]:
    """
    recent + older filings JSON files까지 모두 합쳐서 반환
    """
    all_filings = []

    # 1) recent
    recent_block = submissions_json.get("filings", {}).get("recent", {})
    all_filings.extend(parse_filings_columnar(recent_block))

    # 2) older files
    older_files = submissions_json.get("filings", {}).get("files", [])
    for file_info in older_files:
        name = file_info.get("name")
        if not name:
            continue

        # 보통 submissions/CIKxxxx-submissions-001.json 같은 형태
        if name.startswith("/"):
            url = f"{BASE_DATA_URL}{name}"
        else:
            url = f"{BASE_DATA_URL}/submissions/{name}"

        try:
            older_json = fetch_json(url, use_data_headers=True)
            all_filings.extend(parse_filings_columnar(older_json))
        except Exception as e:
            print(f"[WARN] Failed to load older submissions file: {url} | {e}")

        time.sleep(0.2)

    # 중복 제거
    dedup = {}
    for f in all_filings:
        dedup[f["accession_number"]] = f

    return list(dedup.values())


def filter_filings_by_date(
    filings: List[Dict],
    target_date: str,
    form_types: Optional[List[str]] = None,
    limit: int = 10,
    prefer_on_or_before: bool = True,
) -> List[Dict]:
    """
    target_date와 가까운 filing을 고르되,
    기본적으로는 target_date 이전(또는 당일) filings를 우선
    """
    if form_types is None:
        form_types = ["10-Q", "10-K", "8-K"]

    target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

    candidates = []
    for f in filings:
        if f["form"] not in form_types:
            continue
        try:
            filing_dt = datetime.strptime(f["filing_date"], "%Y-%m-%d").date()
        except Exception:
            continue

        delta_days = (target_dt - filing_dt).days

        # 기본은 target_date 이전/당일 filings만
        if prefer_on_or_before and delta_days < 0:
            continue

        candidates.append((abs(delta_days), delta_days, filing_dt, f))

    # 가까운 날짜 순, 같은 거리면 더 최근 filing 우선
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [item[3] for item in candidates[:limit]]


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
    raw_text = re.sub(
        r"<SEC-HEADER>.*?</SEC-HEADER>",
        " ",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    raw_text = re.sub(
        r"begin\s+\d{3}\s+.*?\n.*?\nend",
        " ",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text_match = re.search(
        r"<TEXT>(.*)</TEXT>",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if text_match:
        raw_text = text_match.group(1)

    raw_text = re.sub(r"</?XBRL[^>]*>", " ", raw_text, flags=re.IGNORECASE)

    soup = BeautifulSoup(raw_text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"<[^>]+>", " ", text)

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

        if sum(p in lower for p in noisy_patterns) >= 2:
            continue

        if len(line) <= 3 and lower in {"true", "false", "—"}:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.split(r"\nEX-\d+(\.\d+)?\n", text, maxsplit=1, flags=re.IGNORECASE)[0]
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


def fetch_sec_filings_for_ticker_near_date(
    ticker: str,
    target_date: str,
    form_types: Optional[List[str]] = None,
    limit: int = 3,
    sleep_sec: float = 0.2,
) -> List[Dict]:
    """
    target_date 근처(기본적으로 target_date 이전/당일)의 filings를 가져옴
    """
    cik = find_cik_by_ticker(ticker)
    if not cik:
        raise ValueError(f"CIK not found for ticker: {ticker}")

    submissions = get_submissions_json(cik)
    all_filings = get_all_filings(submissions)

    selected = filter_filings_by_date(
        all_filings,
        target_date=target_date,
        form_types=form_types or ["10-Q", "10-K", "8-K"],
        limit=limit,
        prefer_on_or_before=True,
    )

    print(f"[INFO] Selected filings near {target_date} for {ticker}:")
    for f in selected:
        print(f"  - {f['form']} {f['filing_date']} {f['accession_number']}")

    results = []
    for filing in selected:
        item = fetch_one_filing_txt_only(cik, ticker, filing)
        if item is not None:
            results.append(item)
        time.sleep(sleep_sec)

    return results


def main():
    ticker = input("Ticker를 입력하세요 (예: AAPL): ").strip().upper()
    target_date = input("기준 날짜를 입력하세요 (예: 2016-07-26): ").strip()

    results = fetch_sec_filings_for_ticker_near_date(
        ticker=ticker,
        target_date=target_date,
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