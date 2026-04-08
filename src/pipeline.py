# src/pipeline.py
from claim_extraction import load_transcript, extract_claims_with_llama
from retrieve import retrieve_top_chunks_for_claim
from query_parser import parse_user_query
from event_resolver import resolve_transcript
from report_generator import generate_report


def select_relevant_claims(claims, user_query, top_n=3):
    """
    임시 버전:
    user_query 단어와 겹치는 claim 우선 선택
    나중에는 embedding 기반 claim retrieval로 개선 가능
    """
    query_words = set(user_query.lower().split())

    scored = []
    for claim in claims:
        claim_text = claim["claim"].lower()
        overlap = sum(1 for w in query_words if w in claim_text)
        scored.append((overlap, claim))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [c for score, c in scored[:top_n] if score > 0]

    if not selected:
        selected = claims[:top_n]

    return selected


def main():
    user_query = input("질문을 입력하세요: ").strip()

    if not user_query:
        print("질문이 비어 있습니다.")
        return

    query_info = parse_user_query(user_query)

    if not query_info["company"]:
        print("회사명을 인식하지 못했습니다. 예: apple, tesla")
        return

    event_info = resolve_transcript(query_info)
    if not event_info:
        print("해당 조건에 맞는 transcript를 찾을 수 없습니다.")
        return

    print(f"\n[INFO] Selected transcript: {event_info['path']}")

    transcript_text = load_transcript(event_info["path"])
    claims = extract_claims_with_llama(transcript_text)

    if not claims:
        print("No claims extracted.")
        return

    relevant_claims = select_relevant_claims(claims, user_query, top_n=3)

    analyzed_claims = []
    for claim in relevant_claims:
        retrieved = retrieve_top_chunks_for_claim(claim["claim"], top_k=2)
        analyzed_claims.append(
            {
                "claim": claim,
                "evidence": retrieved,
            }
        )

    report = generate_report(user_query, event_info, analyzed_claims)
    print("\n" + report)


if __name__ == "__main__":
    main()