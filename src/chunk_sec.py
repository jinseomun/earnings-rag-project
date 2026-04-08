# src/chunk_sec.py
import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 150


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def chunk_text(text: str):
    text = text.strip()
    if not text:
        return []
    return splitter.split_text(text)

RAW_SEC_TEXT_DIR = Path("data/raw/sec/text")
OUTPUT_FILE = Path("data/processed/sec_chunks.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 150


def load_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_sec_filename(file_path: Path) -> dict:
    """
    파일명 예시:
    aapl_2026-01-30_10-Q_clean.txt
    """
    stem = file_path.stem  # aapl_2026-01-30_10-Q_clean

    match = re.match(r"^([a-zA-Z0-9]+)_(\d{4}-\d{2}-\d{2})_([A-Za-z0-9\-]+)_clean$", stem)
    if not match:
        return {
            "ticker": None,
            "filing_date": None,
            "form_type": None,
            "source_file": file_path.name,
        }

    ticker, filing_date, form_type = match.groups()

    return {
        "ticker": ticker.upper(),
        "filing_date": filing_date,
        "form_type": form_type.upper(),
        "source_file": file_path.name,
    }



def chunk_text(text: str):
    text = text.strip()
    if not text:
        return []
    return splitter.split_text(text)


def build_sec_chunks():
    all_chunks = []
    txt_files = sorted(RAW_SEC_TEXT_DIR.glob("*.txt"))

    if not txt_files:
        print(f"No txt files found in {RAW_SEC_TEXT_DIR}")
        return []

    chunk_id = 1

    for file_path in txt_files:
        metadata = parse_sec_filename(file_path)
        text = load_text(file_path)
        chunks = chunk_text(text)

        print(f"[INFO] {file_path.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks, start=1):
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "ticker": metadata["ticker"],
                    "filing_date": metadata["filing_date"],
                    "form_type": metadata["form_type"],
                    "source_file": metadata["source_file"],
                    "chunk_index": i,
                }
            )
            chunk_id += 1

    return all_chunks


def save_chunks(chunks, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    chunks = build_sec_chunks()

    if not chunks:
        print("No chunks created.")
        return

    save_chunks(chunks, OUTPUT_FILE)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()