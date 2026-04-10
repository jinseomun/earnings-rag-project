# src/judge_report.py
import json
from typing import Any, Dict


REPORT_JUDGE_PROMPT = """You are evaluating a generated financial analysis report.

Instructions:
- Judge ONLY based on the claim, evidence, and generated report.
- Do NOT use outside knowledge.
- Be strict.
- Check whether the report is grounded in the provided evidence.
- Mark hallucination as true if the report introduces unsupported factual content.

Return JSON only in this format:
{{
  "faithfulness": "high" | "medium" | "low",
  "hallucination": true,
  "relevance_score": 1,
  "reason": "short explanation"
}}

Claim:
{claim}

Evidence:
{evidence}

Generated Report:
{report}
"""


def build_report_prompt(claim: str, evidence: str, report: str) -> str:
    return REPORT_JUDGE_PROMPT.format(
        claim=claim,
        evidence=evidence,
        report=report,
    )


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


def judge_report_with_openai(client, model: str, claim: str, evidence: str, report: str) -> Dict[str, Any]:
    prompt = build_report_prompt(claim, evidence, report)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict judge for grounded financial report generation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    return parse_json_response(content)