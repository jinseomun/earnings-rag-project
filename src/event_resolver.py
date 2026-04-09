import json
from pathlib import Path
from datetime import datetime

EMBED_FILE = Path("data/processed/earnings_call_chunks_train.json")


def infer_quarter_from_call_date(date_str: str):
    """
    earnings call date 기준으로 fiscal quarter를 대략 추정
    예:
    2017-01 ~ 2017-03  -> 2016Q4
    2017-04 ~ 2017-06  -> 2017Q1
    2017-07 ~ 2017-09  -> 2017Q2
    2017-10 ~ 2017-12  -> 2017Q3
    """
    if not date_str:
        return None

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    y, m = dt.year, dt.month

    if 1 <= m <= 3:
        return f"{y-1}Q4"
    elif 4 <= m <= 6:
        return f"{y}Q1"
    elif 7 <= m <= 9:
        return f"{y}Q2"
    else:
        return f"{y}Q3"


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
