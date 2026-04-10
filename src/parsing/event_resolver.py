import json
from pathlib import Path
from datetime import datetime

EMBED_FILE = Path("data/processed/earnings_call_chunks_train.json")


def infer_quarter_from_call_date(date_str: str):
    """
    call date를 달력 기준 quarter로 매핑
    예:
    2016-01 ~ 2016-03 -> 2016Q1
    2016-04 ~ 2016-06 -> 2016Q2
    2016-07 ~ 2016-09 -> 2016Q3
    2016-10 ~ 2016-12 -> 2016Q4
    """
    if not date_str:
        return None

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    y, m = dt.year, dt.month

    if 1 <= m <= 3:
        return f"{y}Q1"
    elif 4 <= m <= 6:
        return f"{y}Q2"
    elif 7 <= m <= 9:
        return f"{y}Q3"
    else:
        return f"{y}Q4"

def list_events():
    if not EMBED_FILE.exists():
        print("EMBED_FILE not found:", EMBED_FILE.resolve())
        return []

    with open(EMBED_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    print("total rows:", len(rows))

    seen = {}
    for row in rows:
        md = row.get("metadata", {})
        company = (md.get("company") or "").lower()
        ticker = (md.get("ticker") or "").lower()
        call_id = md.get("call_id")
        date = md.get("date")

        if not company or not call_id or not date:
            continue

        inferred_quarter = infer_quarter_from_call_date(date)

        if call_id not in seen:
            seen[call_id] = {
                "company": company,
                "ticker": ticker,
                "quarter": inferred_quarter,
                "call_id": call_id,
                "date": date,
            }

    print("events found:", len(seen))
    return list(seen.values())


def resolve_transcript(query_info: dict):
    # 입력 값이 대소문자/공백 때문에 매칭되지 않는 경우를 방지한다.
    company = (query_info.get("company") or "").strip().lower()
    ticker = (query_info.get("ticker") or "").strip().lower()
    target_quarter = (query_info.get("quarter") or "").strip().upper()
    print(
        f"Resolving transcript for company='{company}', ticker='{ticker}', quarter='{target_quarter}'"
    )


    events = list_events()
    print(events[:3])  # debug: 처음 3개 이벤트 출력
    print(f"[debug] total events: {len(events)}")

    if company:
        events = [e for e in events if e["company"] == ticker]
        print(f"[debug] after company filter ({company}): {len(events)}")

    if ticker:
        events = [e for e in events if e.get("ticker") == ticker]
        print(f"[debug] after ticker filter ({ticker}): {len(events)}")

    if not events:
        print("[debug] no events after filters")
        return None

    if target_quarter:
        exact = [e for e in events if e.get("quarter") == target_quarter]
        print(f"[debug] after quarter filter ({target_quarter}): {len(exact)}")
        if exact:
            return exact[0]
        else:
            print("[debug] no exact quarter match, fallback to latest event")

    # fallback: 최신 날짜
    events.sort(key=lambda x: x.get("date") or "", reverse=True)
    return events[0]
