# src/report_generator.py
def generate_report(user_query, event_info, analyzed_claims):
    lines = []
    lines.append("=== Earnings Call Analysis Report ===")
    lines.append(f"User Query: {user_query}")
    lines.append(f"Company: {event_info['company']}")
    lines.append(f"Quarter: {event_info['quarter']}")
    lines.append(
        f"Transcript: call_id={event_info.get('call_id')} date={event_info.get('date')}"
    )
    lines.append("")

    for idx, item in enumerate(analyzed_claims, start=1):
        claim = item["claim"]
        evidence = item["evidence"]

        lines.append(f"[Claim {idx}]")
        lines.append(f"Text: {claim['claim']}")
        lines.append(f"Speaker: {claim.get('speaker', 'Unknown')}")
        lines.append("Evidence:")

        if not evidence:
            lines.append("- No evidence retrieved.")
        else:
            for ev in evidence:
                score = ev.get("score", 0.0)
                text = ev.get("text", "").replace("\n", " ").strip()
                lines.append(f"- score={score:.4f}")
                lines.append(f"  {text[:300]}")

        lines.append("-" * 60)

    return "\n".join(lines)
