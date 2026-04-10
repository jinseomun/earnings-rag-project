def generate_report(user_query, event_info, analyzed_claims):
    lines = []
    lines.append("=== Earnings Call Analysis Report ===")
    lines.append(f"User Query: {user_query}")
    lines.append(f"Company: {event_info.get('company')}")
    lines.append(f"Quarter: {event_info.get('quarter')}")
    lines.append(
        f"Transcript: call_id={event_info.get('call_id')} date={event_info.get('date')}"
    )
    lines.append("")

    for idx, item in enumerate(analyzed_claims, start=1):
        claim = item["claim"]
        evidence_list = item["evidence"]

        lines.append(f"[Claim {idx}]")
        lines.append(f"Text: {claim.get('claim')}")
        lines.append(f"Speaker: {claim.get('speaker', 'Unknown')}")
        lines.append("Evidence:")

        for ev in evidence_list:
            retrieval_score = ev.get("retrieval_score", 0.0)
            rerank_score = ev.get("rerank_score", 0.0)
            text = ev.get("text", "")

            lines.append(
                f"- retrieval_score={retrieval_score:.4f} | rerank_score={rerank_score:.4f}"
            )
            lines.append(f"  {text[:400]}")

        lines.append("-" * 48)

    return "\n".join(lines)