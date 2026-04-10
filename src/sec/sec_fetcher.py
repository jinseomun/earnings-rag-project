# src/sec_fetcher.py
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
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
    all_filings = []

    recent_block = submissions_json.get("filings", {}).get("recent", {})
    all_filings.extend(parse_filings_columnar(recent_block))

    older_files = submissions_json.get("filings", {}).get("files", [])
    for file_info in older_files:
        name = file_info.get("name")
        if not name:
            continue

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

    dedup = {}
    for f in all_filings:
        dedup[f["accession_number"]] = f

    return list(dedup.values())


def is_earnings_related_filing(filing: Dict) -> bool:
    text = f"{filing.get('primary_document', '')} {filing.get('primary_doc_description', '')}".lower()

    keywords = [
        "results of operations",
        "financial results",
        "earnings",
        "quarterly results",
        "quarter ended",
        "press release",
        "data sheet",
        "exhibit 99",
        "item 2.02",
    ]
    return any(k in text for k in keywords)


def filter_filings_by_date(
    filings: List[Dict],
    target_date: str,
    form_types: Optional[List[str]] = None,
    limit: int = 10,
    prefer_on_or_before: bool = True,
) -> List[Dict]:
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

        if prefer_on_or_before and delta_days < 0:
            continue

        earnings_bonus = 0
        if is_earnings_related_filing(f):
            earnings_bonus = -1000

        candidates.append((earnings_bonus, abs(delta_days), delta_days, filing_dt, f))

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return [item[4] for item in candidates[:limit]]


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


def is_candidate_document(filename: str) -> bool:
    name = filename.lower()

    allowed_ext = (".htm", ".html", ".txt")
    if not name.endswith(allowed_ext):
        return False

    bad_keywords = [
        ".xml",
        ".xsd",
        ".xls",
        ".xlsx",
        ".zip",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".pdf",
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
        "graphic",
        "image",
    ]
    if any(bad in name for bad in bad_keywords):
        return False

    return True


def classify_archive_item(item_name: str, filing: Dict) -> Tuple[str, int]:
    """
    return: (doc_role, score)
    doc_role:
      - main
      - exhibit_99_1
      - exhibit_99_2
      - exhibit_99_other
      - other
    """
    name = item_name.lower()
    primary = filing.get("primary_document", "").lower()
    primary_base = primary.rsplit(".", 1)[0] if "." in primary else primary
    name_base = name.rsplit(".", 1)[0] if "." in name else name

    score = 0
    doc_role = "other"

    if name == primary:
        score += 200
        doc_role = "main"
    elif name_base == primary_base:
        score += 150
        doc_role = "main"

    # exhibit 99.x 쪽 강한 우선순위
    exhibit_patterns_99_1 = [
        "99.1", "99_1", "99-1", "ex99_1", "ex99-1", "ex991", "ex-99.1"
    ]
    exhibit_patterns_99_2 = [
        "99.2", "99_2", "99-2", "ex99_2", "ex99-2", "ex992", "ex-99.2"
    ]
    exhibit_patterns_99 = [
        "99.", "99_", "99-", "ex99", "ex-99", "exhibit99"
    ]

    if any(p in name for p in exhibit_patterns_99_1):
        score += 180
        doc_role = "exhibit_99_1"
    elif any(p in name for p in exhibit_patterns_99_2):
        score += 170
        doc_role = "exhibit_99_2"
    elif any(p in name for p in exhibit_patterns_99):
        score += 120
        doc_role = "exhibit_99_other"

    # earnings 관련 키워드
    positive_keywords = [
        "press",
        "release",
        "earnings",
        "results",
        "quarter",
        "financial",
        "datasheet",
        "data-sheet",
        "factsheet",
        "factsheet",
    ]
    for kw in positive_keywords:
        if kw in name:
            score += 20

    # 불필요 문서 감점
    negative_keywords = [
        "10.1",
        "4.1",
        "4.2",
        "4.3",
        "4.4",
        "5.1",
        "12.1",
        "23.1",
        "underwriting",
        "indenture",
        "note",
        "officer",
        "opinion",
        "consent",
        "fixedcharges",
        "ratio",
    ]
    for kw in negative_keywords:
        if kw in name:
            score -= 40

    if name.endswith(".htm") or name.endswith(".html"):
        score += 10

    if name.endswith(".txt"):
        score += 5

    return doc_role, score


def choose_relevant_documents(
    filing: Dict,
    archive_items: List[dict],
    max_docs: int = 5,
) -> List[Dict]:
    candidates = []

    for item in archive_items:
        name = item.get("name", "")
        if not name or not is_candidate_document(name):
            continue

        doc_role, score = classify_archive_item(name, filing)
        candidates.append(
            {
                "name": name,
                "doc_role": doc_role,
                "score": score,
            }
        )

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    seen_names = set()
    seen_roles = set()

    for c in candidates:
        name = c["name"]
        role = c["doc_role"]

        if name in seen_names:
            continue

        # main, ex99_1, ex99_2는 가능하면 각각 하나씩 챙김
        if role in {"main", "exhibit_99_1", "exhibit_99_2"}:
            if role in seen_roles:
                continue
            selected.append(c)
            seen_names.add(name)
            seen_roles.add(role)
            continue

    # 아직 너무 적으면 점수 높은 나머지 추가
    for c in candidates:
        if len(selected) >= max_docs:
            break
        if c["name"] in seen_names:
            continue
        if c["score"] < 40:
            continue
        selected.append(c)
        seen_names.add(c["name"])

    return selected[:max_docs]


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

    # 예전처럼 EX- 뒤를 무조건 잘라버리지 않음
    # exhibit 본문이 실제로 필요한 경우가 있으므로 삭제

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


def fetch_one_filing_with_exhibits(cik: str, ticker: str, filing: Dict) -> List[Dict]:
    accession_number = filing["accession_number"]

    index_json = fetch_archive_index_json(cik, accession_number)
    archive_items = get_archive_items(index_json) if index_json else []

    chosen_docs = choose_relevant_documents(filing, archive_items, max_docs=5)

    if not chosen_docs:
        print(
            f"[SKIP] {filing['form']} {filing['filing_date']} | "
            f"No usable filing/exhibit documents found."
        )
        return []

    print(
        f"[INFO] {filing['form']} {filing['filing_date']} | "
        f"selected docs: {[d['name'] for d in chosen_docs]}"
    )

    results = []

    for doc in chosen_docs:
        source_filename = doc["name"]
        doc_role = doc["doc_role"]

        source_url = build_filing_document_url(cik, accession_number, source_filename)

        try:
            raw_text = download_text(source_url)
            clean_text = clean_sec_filing_text(raw_text)

            if len(clean_text.strip()) < 200:
                print(f"[SKIP] too short after cleaning: {source_filename}")
                continue

            suffix = doc_role
            text_path = save_clean_text(ticker, filing, clean_text, suffix)

            results.append(
                {
                    "ticker": ticker.upper(),
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "accession_number": accession_number,
                    "doc_role": doc_role,
                    "source_filename": source_filename,
                    "source_url": source_url,
                    "clean_text_path": str(text_path),
                }
            )

        except Exception as e:
            print(f"[WARN] Failed to fetch {source_filename} | {e}")

        time.sleep(0.2)

    return results


def fetch_sec_filings_for_ticker_near_date(
    ticker: str,
    target_date: str,
    form_types: Optional[List[str]] = None,
    limit: int = 3,
    sleep_sec: float = 0.2,
) -> List[Dict]:
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
        desc = f.get("primary_doc_description", "")
        print(f"  - {f['form']} {f['filing_date']} {f['accession_number']} | {desc}")

    results = []
    for filing in selected:
        items = fetch_one_filing_with_exhibits(cik, ticker, filing)
        results.extend(items)
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

    print("\n=== Saved SEC Documents ===")
    if not results:
        print("No usable filings/exhibits found.")
        return

    for item in results:
        print(item)


if __name__ == "__main__":
    main()