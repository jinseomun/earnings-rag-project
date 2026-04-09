import json
from pathlib import Path

EMBED_FILE = Path("data/processed/earnings_call_embeddings_train.json")


def load_transcript_from_json(call_id: str) -> str:
    if not EMBED_FILE.exists():
        print("Embedding file not found.")
        return ""

    with open(EMBED_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    matched = []
    for row in rows:
        md = row.get("metadata", {})
        if md.get("call_id") == call_id:
            matched.append(row)

    if not matched:
        print(f"No chunks found for call_id: {call_id}")
        return ""

    # 순서 복원 (중요)
    matched.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))

    # 다시 하나의 transcript로 합침
    return "\n\n".join(row["page_content"] for row in matched)