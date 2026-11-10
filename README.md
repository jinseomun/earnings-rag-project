# Earnings Call Fact-checking with RAG

기업의 Earnings Call Transcript에서 재무 관련 Claim을 추출하고, SEC Filings에서 해당 Claim을 검증할 Evidence를 검색하는 RAG 기반 분석 프로젝트입니다. Dense Retrieval과 Cross-encoder Reranking으로 Evidence를 선별하고, LLM-as-a-Judge로 검색 및 분석 결과를 평가합니다.

## 주요 기능

- **Claim Extraction:** Llama 3.1 8B Instruct로 Earnings Call Transcript에서 매출, 이익, 성장률, 가이던스 등 검증 가능한 발언 추출
- **SEC Evidence Retrieval:** 기업과 발표 시점에 맞는 SEC 공시 수집 및 E5 기반 의미 검색
- **Reranking:** 재무 지표·숫자·비교 표현을 고려한 후보 점수 보정과 MiniLM Cross-encoder 재정렬
- **Claim-level Report:** Claim별 Evidence 텍스트 일부와 Retrieval·Reranking 점수를 구조화된 텍스트로 출력
- **Evaluation:** GPT-4.1-mini를 이용한 Evidence Support 판정 및 Report Faithfulness·Hallucination·Relevance 평가

## 시스템 구조

```text
사용자 질문 (기업 / 분기)
         │
         ▼
Earnings Call Transcript 선택
         │
         ▼
Llama 3.1 8B Instruct → Claim 추출
         │
         ├───────────────────────────────┐
         │                               │
         │                  SEC EDGAR 공시 수집
         │                               ▼
         │                       Chunking + E5 Embedding
         │                               │
         ▼                               ▼
    Claim 기반 Query ────────────► Dense Retrieval
                                         │
                              재무 도메인 점수 보정
                                         │
                              MiniLM Cross-encoder
                                    Reranking
                                         │
                                         ▼
                              Claim별 Evidence 출력
                                         │
                                         ▼
                              Retrieval / Report 평가
```

### 데이터 및 모델

| 구성 요소 | 사용 기술 / 설정 |
|---|---|
| Earnings Call Transcript | Hugging Face `jlh-ibm/earnings_call` |
| 재무 공시 | SEC EDGAR (10-K, 10-Q, 8-K 및 첨부 자료) |
| Claim Extraction | `meta-llama/Llama-3.1-8B-Instruct` |
| Dense Retriever | `intfloat/e5-base-v2` |
| Cross-encoder Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| SEC Chunking | `RecursiveCharacterTextSplitter` (700 / 100 overlap) |
| Transcript Chunking | `RecursiveCharacterTextSplitter` (1,200 / 150 overlap) |
| Evaluation | GPT-4.1-mini (LLM-as-a-Judge) |

SEC 공시 문서는 텍스트로 변환한 뒤 청크 단위로 분할합니다. E5 임베딩으로 후보를 검색한 후, 관련 재무 지표·숫자·비교 표현에 점수 보정을 적용합니다. Cross-encoder에는 Claim과 후보 청크의 텍스트를 함께 입력해 관련성 점수를 다시 계산하고 순위를 재정렬합니다.

## 실험 결과

8개의 재무 Claim을 대상으로 Retrieval 성능과 최종 분석 결과를 평가했습니다.

### Retrieval Evaluation

| 지표 | 결과 |
|---|---:|
| Hit@1 | 0.2500 |
| Hit@3 | 0.6250 |
| Partial Hit@3 | 0.6250 |
| MRR | 0.4375 |
| Precision@3 | 0.2500 |

### Report Evaluation

| 지표 | 결과 |
|---|---:|
| High Faithfulness Rate | 0.3750 |
| Hallucination Rate | 0.6250 |
| Average Relevance Score | 0.6250 |

평가 모델은 Claim과 검색된 Evidence를 비교하여 `supported`, `partially_supported`, `not_supported`로 판정합니다. 보고서 평가는 Faithfulness, Hallucination, Relevance를 별도로 평가합니다.

### Ablation Study

SEC Evidence Limit과 Reranker 적용 여부를 변경해 검색 결과를 비교했습니다.

| 설정 | Hit@1 | Hit@3 | MRR | Hallucination Rate |
|---|---:|---:|---:|---:|
| 기본 설정 (SEC Limit 3) | 0.2500 | 0.6250 | 0.4375 | 0.6250 |
| SEC Limit 4 | 0.3750 | 0.6250 | 0.5250 | 0.7500 |
| Reranker 제거 | 0.1250 | 0.5000 | 0.3125 | 0.6250 |

- **Evidence 수 증가:** SEC Limit을 4로 높이면 Hit@1과 MRR은 개선되었지만, Hallucination Rate도 증가했습니다. Evidence 수를 늘리는 것이 항상 Report의 Faithfulness를 개선하지는 않았습니다.
- **Reranker 제거:** Hit@1, Hit@3, MRR이 모두 감소했습니다. Cross-encoder 재정렬이 상위 Retrieval 결과의 Ranking 품질에 기여했음을 보여줍니다.

실험은 8개 Claim을 대상으로 수행했습니다.

## 프로젝트 구조

```text
earnings-rag-project/
├── data/
│   ├── raw/sec/text/                  # SEC 공시 텍스트
│   └── processed/                    # 정제 데이터, 청크, 임베딩
├── src/
│   ├── claims/                       # Earnings Call 로드, 청크, Claim 추출
│   ├── sec/                          # SEC 공시 수집, 청크, 임베딩
│   ├── parsing/                      # 기업·분기 파싱 및 Transcript 탐색
│   ├── retrieval/retrieve.py         # Dense Retrieval 및 Reranking
│   ├── reports/report_generator.py   # Evidence 기반 결과 출력
│   ├── pipeline/pipeline.py          # 메인 파이프라인
│   ├── evaluate_pipeline.py          # 평가 실행
│   ├── judge_evidence.py             # Evidence 판정
│   ├── judge_report.py               # Report 평가
│   └── eval_utils.py                 # 평가 유틸리티
├── outputs/
│   ├── claims.json
│   ├── retrieval_eval_results.json
│   └── report_eval_results.json
└── requirements.txt
```

## 실행 방법

### 1. 설치 및 환경 변수

```bash
git clone https://github.com/jinseomun/earnings-rag-project.git
cd earnings-rag-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

프로젝트 루트에 `.env`를 생성합니다.

```dotenv
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

SEC 요청 전 `src/sec/sec_fetcher.py`의 `User-Agent`를 자신의 연락처 정보로 설정합니다.

### 2. Earnings Call 데이터 전처리

```bash
python src/claims/load_earnings_call.py
python src/claims/chunk_earnings_call.py
```

Transcript용 FAISS 인덱스가 필요한 경우 별도 임베딩 스크립트를 실행합니다.

```bash
python src/claims/embed_earnings_call.py
```

### 3. 분석 파이프라인 실행

```bash
python src/pipeline/pipeline.py
```

기업과 분기가 포함된 질문을 입력하면 관련 Earnings Call을 선택하고 SEC Evidence를 검색하여 Claim별 분석 결과를 출력합니다.

### 4. 평가 실행

`outputs/claims.json`과 `data/processed/sec_embeddings.json`이 준비된 상태에서 실행합니다.

```bash
PYTHONPATH=src python src/evaluate_pipeline.py
```

평가 결과는 `outputs/retrieval_eval_results.json`과 `outputs/report_eval_results.json`에 저장됩니다.

## 한계 및 개선 방향

- 숫자가 일치하더라도 기간, 비교 기준, 가이던스 여부가 다르면 충분한 Evidence가 아닐 수 있습니다.
- SEC 표를 텍스트로 변환할 때 값과 열 제목의 대응 관계가 손실될 수 있습니다.
- Evidence를 많이 제공하면 관련성이 낮은 문서까지 포함되어 결과 품질이 저하될 수 있습니다.
- 더 다양한 기업·분기·공시 유형을 대상으로 평가하고, Retrieval 성능과 Evidence-grounded Generation의 Faithfulness를 함께 개선할 필요가 있습니다.
