import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") #토큰 불러옴
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" #불러올 모델 이름 설정 (변경 가능)

RAW_TRANSCRIPT_DIR = Path("data/raw/transcript") #원본 transcript이 저장된 디렉토리
OUTPUT_DIR = Path("outputs") #추출된 claim이 저장될 디렉토리
OUTPUT_DIR.mkdir(exist_ok=True) #출력 디렉토리가 없으면 생성, 이미 있으면 그냥 넘어감

client = InferenceClient( #허깅페이스 서버에 요청 보내는 클라이언트 초기화
    provider="auto",
    api_key=HF_TOKEN,
)


def load_transcript(file_path: Path) -> str: #원본 transcript 읽어오기
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_json_block(text: str): #json 블록만 추출
    """
    모델 출력에서 JSON 배열만 최대한 안전하게 추출
    """
    match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if not match:
        return None
    return match.group(0)


def extract_claims_with_llama(transcript_text: str): #시스템 프롬프트
    system_prompt = (
        "You are a financial analyst assistant. "
        "Extract only the most important factual, specific, verifiable claims from an earnings call transcript. "
        "Prioritize financially material claims that are likely to be checked against SEC filings, "
        "such as revenue, profit, EPS, margins, guidance, segment performance, and major growth or decline metrics. "
        "Ignore vague, repetitive, promotional, or non-material statements. "
        "Return only a valid JSON array. "
        "Do not include markdown fences or any explanation."
    )

    user_prompt = f"""
        Extract the most important factual claims from the following earnings call transcript.

        Rules:
        1. Select at most 8 claims.
        2. Each claim must be specific, concise, and verifiable.
        3. Prioritize financially material claims:
        - revenue
        - net income
        - EPS
        - gross margin
        - operating expenses
        - guidance
        - segment performance (e.g. iPhone, Services)
        - major growth/decline percentages
        4. Avoid vague, generic, promotional, or repetitive statements.
        5. Include speaker when identifiable.
        6. Return ONLY valid JSON.
        7. Do not include markdown fences.
        8. If multiple claims say nearly the same thing, keep only one.

        Output format:
        [
        {{
            "claim_id": 1,
            "claim": "Revenue was $143.8 billion, up 16% year-over-year.",
            "speaker": "Kevan Parekh"
        }}
        ]

        Transcript:
        {transcript_text}
        """

    completion = client.chat.completions.create( #실제 요청 보내는 부분
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    raw_output = completion.choices[0].message.content

    json_block = extract_json_block(raw_output)
    if json_block is None:
        print("JSON block not found.")
        print("Raw output:\n", raw_output)
        return []

    try:
        claims = json.loads(json_block)
    except json.JSONDecodeError:
        print("JSON parsing failed.")
        print("Raw output:\n", raw_output)
        return []

    return claims


def save_claims(claims, output_path: Path): #추출된 claim을 json 파일로 저장 (outputs/claims.json)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(claims, f, ensure_ascii=False, indent=2)


def main():
    transcript_file = RAW_TRANSCRIPT_DIR / "apple_2026Q1.txt"
    output_file = OUTPUT_DIR / "claims.json"

    text = load_transcript(transcript_file)
    claims = extract_claims_with_llama(text)
    save_claims(claims, output_file)

    print(f"Saved {len(claims)} claims to {output_file}")


if __name__ == "__main__":
    main()