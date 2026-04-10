import json
from pathlib import Path
from datasets import load_dataset

# HuggingFace parquet 경로
TRAIN_URL = "https://huggingface.co/datasets/jlh-ibm/earnings_call/resolve/refs%2Fconvert%2Fparquet/transcripts/train/*.parquet"
TEST_URL = "https://huggingface.co/datasets/jlh-ibm/earnings_call/resolve/refs%2Fconvert%2Fparquet/transcripts/test/*.parquet"

# 저장 경로
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_row(row: dict, split: str) -> dict | None:
    """
    각 row를 RAG용 document 형태로 변환
    """

    # transcript 본문
    text = row.get("text") or row.get("transcript") or row.get("content")
    if not text or not str(text).strip():
        return None

    # 메타데이터 추출
    company = row.get("company") or row.get("symbol") or row.get("ticker")
    date = row.get("date") or row.get("call_date") or row.get("timestamp")
    quarter = row.get("quarter") or row.get("fiscal_quarter")
    year = row.get("year") or row.get("fiscal_year")
    title = row.get("title")
    ticker = row.get("ticker") or row.get("symbol") or company

    # 🔥 JSON 저장 가능하게 전부 문자열 처리
    company = str(company) if company is not None else None
    ticker = str(ticker) if ticker is not None else None
    date = str(date) if date is not None else None
    quarter = str(quarter) if quarter is not None else None
    year = str(year) if year is not None else None
    title = str(title) if title is not None else None

    # call_id 생성
    call_id_parts = [
        x for x in [ticker, year, quarter, date]
        if x not in (None, "", "nan")
    ]
    call_id = "_".join(call_id_parts) if call_id_parts else None

    return {
        "page_content": text.strip(),
        "metadata": {
            "doc_type": "earnings_call",
            "split": split,
            "company": company,
            "ticker": ticker,
            "date": date,
            "year": year,
            "quarter": quarter,
            "title": title,
            "call_id": call_id,
        },
    }


def save_json(data: list, path: Path) -> None:
    """
    JSON 저장 (날짜 등 자동 처리)
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def main():
    print("Loading dataset...")

    ds = load_dataset(
        "parquet",
        data_files={
            "train": TRAIN_URL,
            "test": TEST_URL,
        },
    )

    print(ds)

    # -------------------------
    # train 처리
    # -------------------------
    train_docs = []
    for row in ds["train"]:
        doc = normalize_row(row, "train")
        if doc:
            train_docs.append(doc)

    # -------------------------
    # test 처리
    # -------------------------
    test_docs = []
    for row in ds["test"]:
        doc = normalize_row(row, "test")
        if doc:
            test_docs.append(doc)

    # -------------------------
    # 저장
    # -------------------------
    train_path = OUTPUT_DIR / "earnings_calls_train.json"
    test_path = OUTPUT_DIR / "earnings_calls_test.json"

    save_json(train_docs, train_path)
    save_json(test_docs, test_path)

    print(f"\nSaved train docs: {len(train_docs)} → {train_path}")
    print(f"Saved test docs: {len(test_docs)} → {test_path}")

    # -------------------------
    # 샘플 출력
    # -------------------------
    if train_docs:
        print("\nSample metadata:")
        print(train_docs[0]["metadata"])

        print("\nSample text:")
        print(train_docs[0]["page_content"][:500])


if __name__ == "__main__":
    main()


