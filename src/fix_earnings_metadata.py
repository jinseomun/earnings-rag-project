import json
from pathlib import Path
from datetime import datetime

INPUT_FILE = Path("data/processed/earnings_call_embeddings_train.json")
OUTPUT_FILE = Path("data/processed/earnings_call_embeddings_train_fixed.json")


def infer_fiscal_period_from_call_date(date_str: str):
    """
    earnings call date 기준으로 fiscal year / quarter 추정
    예:
    2017-01-xx -> 2016Q4
    2017-04-xx -> 2017Q1
    2017-07-xx -> 2017Q2
    2017-10-xx -> 2017Q3
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    month = dt.month

    if 1 <= month <= 3:
        fiscal_year = year - 1
        fiscal_quarter = "Q4"
    elif 4 <= month <= 6:
        fiscal_year = year
        fiscal_quarter = "Q1"
    elif 7 <= month <= 9:
        fiscal_year = year
        fiscal_quarter = "Q2"
    else:
        fiscal_year = year
        fiscal_quarter = "Q3"

    return fiscal_year, f"{fiscal_year}{fiscal_quarter}"


def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    fixed_count = 0

    for row in rows:
        md = row.get("metadata", {})
        date_str = md.get("date")

        if not date_str:
            continue

        year, quarter = infer_fiscal_period_from_call_date(date_str)
        md["year"] = year
        md["quarter"] = quarter
        fixed_count += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    print(f"Done. Updated {fixed_count} rows.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()