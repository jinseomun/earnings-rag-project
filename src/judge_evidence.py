# src/judge_evidence.py
import json
from typing import Any, Dict


EVIDENCE_JUDGE_PROMPT = """You are evaluating whether a SEC evidence chunk supports a financial claim from an earnings call.

Instructions:
- Judge ONLY based on the claim and the evidence text.
- Do NOT use outside knowledge.
- Be strict.
- Focus on:
  1. financial metric match
  2. numeric value match
  3. unit match (billion, million, %, per share)
  4. comparison match (YoY, QoQ, growth rate)
  5. period match if mentioned
- If the evidence supports only part of the claim, output partially_supported.
- If the evidence is about a different metric, output not_supported.
- Do not assume missing facts.

Return JSON only in this format:
{{
  "label": "supported" | "partially_supported" | "not_supported",
  "metric_match": true,
  "number_match": true,
  "unit_match": true,
  "comparison_match": false,
  "period_match": false,
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
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Failed to parse JSON response: {text}")


def judge_evidence_with_openai(client, model: str, claim: str, evidence: str) -> Dict[str, Any]:
    prompt = build_evidence_prompt(claim, evidence)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict financial evidence judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    return parse_json_response(content)