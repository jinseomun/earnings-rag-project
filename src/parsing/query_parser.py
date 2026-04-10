# src/query_parser.py
import re

KNOWN_COMPANIES = [
    "apple",
    "tesla",
    "microsoft",
    "amazon",
    "meta",
    "google",
    "alphabet",
    "nvidia",
    "intel",
    "amd",
    "micron",
    "cisco",
    "asml",
]

# 회사명/별칭/티커를 표준 티커로 매핑 (event_resolver와 일관 유지)
ALIAS_TICKER = {
    "goog": "googl",
    "google": "googl",
    "alphabet": "googl",
    "aapl": "aapl",
    "apple": "aapl",
    "msft": "msft",
    "microsoft": "msft",
    "amzn": "amzn",
    "amazon": "amzn",
    "meta": "meta",  # meta는 그대로 ticker로 사용
    "fb": "meta",
    "nvda": "nvda",
    "nvidia": "nvda",
    "intc": "intc",
    "intel": "intc",
    "amd": "amd",
    "advanced micro devices": "amd",
    "mu": "mu",
    "micron": "mu",
    "micron technology": "mu",
    "csco": "csco",
    "cisco": "csco",
    "asml": "asml",
    "asml holding": "asml",
    "tsla": "tsla",
    "tesla": "tsla",
}

# 티커 → 대표 회사명(소문자) 역매핑
TICKER_TO_COMPANY = {
    "aapl": "apple",
    "tsla": "tesla",
    "msft": "microsoft",
    "amzn": "amazon",
    "meta": "meta",
    "googl": "alphabet",
    "nvda": "nvidia",
    "intc": "intel",
    "amd": "amd",
    "mu": "micron",
    "csco": "cisco",
    "asml": "asml",
}

TOPIC_KEYWORDS = [
    "ai",
    "guidance",
    "revenue",
    "margin",
    "china",
    "iphone",
    "cloud",
    "services",
    "mac",
    "ipad",
]


def parse_user_query(user_query: str) -> dict:
    q = user_query.lower().strip()

    def normalize_ticker(raw: str) -> str:
        raw = (raw or "").strip().lower()
        if not raw:
            return ""
        # exact ticker pattern 우선
        ticker_match = re.fullmatch(r"[a-z]{1,5}", raw)
        if ticker_match and raw in ALIAS_TICKER.values():
            return raw
        return ALIAS_TICKER.get(raw, raw)

    company = None
    ticker = None

    # 1) 문장 속에 포함된 ticker 패턴 먼저 찾기
    ticker_in_text = re.findall(r"\b([a-z]{1,5})\b", q)
    for t in ticker_in_text:
        nt = normalize_ticker(t)
        if nt in TICKER_TO_COMPANY:
            ticker = nt
            company = TICKER_TO_COMPANY[nt]
            break

    # 2) alias/회사명 스캔
    if not company:
        for alias, tgt in ALIAS_TICKER.items():
            if alias in q:
                ticker = tgt
                company = TICKER_TO_COMPANY.get(tgt, alias)
                break

    quarter_match = re.search(r"(20\d{2})\s*q([1-4])", q)
    if quarter_match:
        quarter = f"{quarter_match.group(1)}Q{quarter_match.group(2)}"
    else:
        quarter_match = re.search(r"(20\d{2}q[1-4])", q)
        quarter = quarter_match.group(1).upper() if quarter_match else None

    date_match = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", q)
    date = None
    if date_match:
        yyyy, mm, dd = date_match.groups()
        date = f"{yyyy}-{int(mm):02d}-{int(dd):02d}"

    latest = any(
        word in q
        for word in [
            "이번 분기",
            "최근",
            "latest",
            "most recent",
            "이번에",
            "이번",
            "recent",
        ]
    )

    topic_keywords = [kw for kw in TOPIC_KEYWORDS if kw in q]

    return {
        "raw_query": user_query,
        "company": company,
        "ticker": ticker,
        "quarter": quarter,
        "date": date,
        "latest": latest,
        "topic_keywords": topic_keywords,
    }
