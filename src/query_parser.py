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
]

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

    company = None
    for c in KNOWN_COMPANIES:
        if c in q:
            company = c
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
        "quarter": quarter,
        "date": date,
        "latest": latest,
        "topic_keywords": topic_keywords,
    }