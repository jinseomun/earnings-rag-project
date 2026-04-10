# src/judge_evidence.py
import json
from typing import Any, Dict


EVIDENCE_JUDGE_PROMPT = """You are evaluating whether an SEC evidence chunk supports a financial claim from an earnings call.

Instructions:
- Judge ONLY based on the claim and the evidence text.
- Do NOT use outside knowledge.
- Be strict but fair.
- Do NOT require exact surface-form number matching.
- If the claim contains a percentage change, growth rate, decline, margin, ratio, or comparison,
  the evidence may still support the claim if that value can be reasonably derived from the numbers in the evidence.
- Allow:
  1. rounding differences
  2. equivalent expressions
  3. table-style evidence
  4. derived values from raw numbers
  5. unit conversion if clearly compatible (for example, millions vs billions)
- Focus on:
  1. financial metric match
  2. whether the numeric claim is directly stated OR derivable
  3. unit match / unit compatibility
  4. comparison match (YoY, QoQ, growth rate, sequential change, decline, basis points)
  5. period match if mentioned
- If the evidence supports only part of the claim, output partially_supported.
- If the evidence is about a different metric, output not_supported.
- If the evidence contradicts the claim, output not_supported.
- Do not assume missing facts.

Label guidance:
- supported:
  The claim is directly stated OR clearly derivable from the evidence.
- partially_supported:
  The evidence is relevant and supports part of the claim, but not enough for full verification.
- not_supported:
  The evidence is irrelevant, insufficient, about a different metric/period, or contradicts the claim.

Important examples:

Example 1
Claim: Revenue increased 10% year-over-year.
Evidence: Revenue was 100 last year and 110 this year.
Output:
{{
  "label": "supported",
  "metric_match": true,
  "number_match": true,
  "unit_match": true,
  "comparison_match": true,
  "period_match": true,
  "derivable": true,
  "reason": "The 10% increase is directly derivable from 100 to 110."
}}

Example 2
Claim: iPhone revenue increased 23% year-over-year.
Evidence: iPhone net sales were 84,492 versus 91,464, change (8)%.
Output:
{{
  "label": "not_supported",
  "metric_match": true,
  "number_match": false,
  "unit_match": true,
  "comparison_match": false,
  "period_match": true,
  "derivable": false,
  "reason": "The evidence shows a decline, not a 23% increase."
}}

Example 3
Claim: Services revenue was $30 billion, up 14% year-over-year.
Evidence: Services net sales were $12,047 million versus $9,795 million, change 23%.
Output:
{{
  "label": "not_supported",
  "metric_match": true,
  "number_match": false,
  "unit_match": true,
  "comparison_match": false,
  "period_match": true,
  "derivable": true,
  "reason": "The evidence is about Services revenue, but the numeric value and growth rate do not match the claim."
}}

Return JSON only in this exact format:
{{
  "label": "supported" | "partially_supported" | "not_supported",
  "metric_match": true,
  "number_match": true,
  "unit_match": true,
  "comparison_match": true,
  "period_match": true,
  "derivable": true,
  "reason": "short explanation"
}}

Claim:
{claim}

Evidence:
{evidence}
"""


def build_evidence_prompt(claim: str, evidence: str) -> str:
    return EVIDENCE_JUDGE_PROMPT.format(claim=claim, evidence=evidence)


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
        else:
            raise ValueError(f"Failed to parse JSON response: {text}")

    # 기본값 보정
    return {
        "label": data.get("label", "not_supported"),
        "metric_match": bool(data.get("metric_match", False)),
        "number_match": bool(data.get("number_match", False)),
        "unit_match": bool(data.get("unit_match", False)),
        "comparison_match": bool(data.get("comparison_match", False)),
        "period_match": bool(data.get("period_match", False)),
        "derivable": bool(data.get("derivable", False)),
        "reason": data.get("reason", ""),
    }


def judge_evidence_with_openai(client, model: str, claim: str, evidence: str) -> Dict[str, Any]:
    prompt = build_evidence_prompt(claim, evidence)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful financial evidence judge. "
                    "You must allow derivable numeric support, not just exact text matches. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    return parse_json_response(content)