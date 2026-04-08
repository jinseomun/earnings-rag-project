# src/event_resolver.py
import re
from pathlib import Path

TRANSCRIPT_DIR = Path("data/raw/transcript")


def list_transcripts():
    results = []

    if not TRANSCRIPT_DIR.exists():
        return results

    for path in TRANSCRIPT_DIR.glob("*.txt"):
        stem = path.stem  # ex) apple_2024Q1
        match = re.match(r"^([a-zA-Z0-9_\-]+)_(20\d{2}Q[1-4]|20\d{2}q[1-4])$", stem)
        if not match:
            continue

        company, quarter = match.groups()
        results.append(
            {
                "company": company.lower(),
                "quarter": quarter.upper(),
                "path": path,
            }
        )

    return results


def quarter_sort_key(quarter: str):
    year = int(quarter[:4])
    q = int(quarter[-1])
    return (year, q)


def resolve_transcript(query_info: dict):
    company = query_info.get("company")
    target_quarter = query_info.get("quarter")

    transcripts = list_transcripts()

    if company:
        transcripts = [t for t in transcripts if t["company"] == company]

    if not transcripts:
        return None

    if target_quarter:
        exact_matches = [t for t in transcripts if t["quarter"] == target_quarter]
        if exact_matches:
            return exact_matches[0]

    transcripts.sort(key=lambda x: quarter_sort_key(x["quarter"]), reverse=True)
    return transcripts[0]