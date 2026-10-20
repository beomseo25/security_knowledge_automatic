# NIST CSF 2.0 → 보안 분류체계 67개 Category 매핑 검증 (Phase D)

NIST CSF 2.0 서브카테고리 22개(영문 한 문장)를 우리 보안 분류체계의 67개 Category 중 하나에 배정하는 실험의
스크립트, 데이터, 결과, 진행 기록을 모았다. 배정 방식은 **BGE-M3 하이브리드 임베딩(dense + sparse + colbert)으로
후보를 정렬하고, 1·2위 점수 차이(마진)가 작은 항목만 LLM이 상위 15개 후보 중에서 다시 고르는** 방식이다.

작성일 2026-10-21. 이 문서의 모든 수치는 `algorithm/results/` 의 결과 CSV에서 `algorithm/summarize_results.py` 로
직접 센 값이다.

## 1. 한눈에 보기

1. **같은 파이프라인이라도 설명서(profile)가 바뀌면 점수가 크게 달라진다.** 한국 인증기준용 deploy profile로는 77.3%(17/22),
   NIST 공개 용어를 덧붙인 profile로는 86.4%(19/22), 오답을 보고 다듬은 profile로는 90.9%(20/22)가 나왔다.
2. **그러나 86.4%, 90.9%는 일반화 성능이 아니다.** [3. 해석할 때 주의](#3-해석할-때-주의)를 먼저 읽을 것.
   profile 작성에 22건을 쓰지 않은 값은 **77.3% 하나뿐**이다.
3. profile 어휘가 가장 크게 움직인 것은 **LLM 이전 단계(임베딩)** 다. 임베딩 hybrid Top-1이 50.0% → 77.3% → 86.4%로 올랐고
   (deploy → NIST 1번 방식 → 3번 방식 r2), LLM 재판정이 더한 정답 수는 +6, +2, +2, +1건(deploy, 1번, r1, r2)으로 줄었다.

## 2. 전체 비교표 (n = 22)

Top-3 열: 임베딩 단독 행은 상위 3개 후보 안에 정답이 있는 비율. **hybrid + LLM 행은** LLM(또는 자동확정)이 정한 최종 답을
1위로 두고, 2~3위를 임베딩 순위에서 겹치지 않는 후보로 채운 값이다. LLM이 하나만 고르기 때문에 이 정의로만 말할 수 있고,
그래서 Top-1보다 늘어난 폭이 작다.

| 조건 | profile | 방식 | Top-1 | Top-3 |
|---|---|---|---|---|
| Phase D | 이름만 | dense | 6/22 (27.3%) | 15/22 (68.2%) |
| 가설 B | 이름만 | hybrid | 8/22 (36.4%) | 15/22 (68.2%) |
| 가설 A | deploy CSV (한국 인증기준용) | dense | 10/22 (45.5%) | 18/22 (81.8%) |
| 가설 A | deploy CSV | hybrid | 11/22 (50.0%) | 15/22 (68.2%) |
| 통합 (CSV) | deploy CSV | hybrid + LLM | 17/22 (77.3%) | 18/22 (81.8%) |
| **최종: Neo4j에서 읽은 profile** | deploy profile (독립 값) | dense | 10/22 (45.5%) | 18/22 (81.8%) |
| | | hybrid | 11/22 (50.0%) | 15/22 (68.2%) |
| | | hybrid + LLM | **17/22 (77.3%)** | 18/22 (81.8%) |
| NIST 공개 용어 profile (1번 방식) | 정답지 영향 가능 | dense | 12/22 (54.5%) | 20/22 (90.9%) |
| | | hybrid | 17/22 (77.3%) | 21/22 (95.5%) |
| | | hybrid + LLM | **19/22 (86.4%)** | 21/22 (95.5%) |
| NIST profile 튜닝 r1 (3번 방식) | in-sample 튜닝 | dense | 14/22 (63.6%) | 20/22 (90.9%) |
| | | hybrid | 17/22 (77.3%) | 21/22 (95.5%) |
| | | hybrid + LLM | **19/22 (86.4%)** | 21/22 (95.5%) |
| NIST profile 튜닝 r2 (3번 방식) | in-sample 튜닝 | dense | 17/22 (77.3%) | 21/22 (95.5%) |
| | | hybrid | 19/22 (86.4%) | 21/22 (95.5%) |
| | | hybrid + LLM | **20/22 (90.9%)** | 21/22 (95.5%) |

- "통합 (CSV)"과 "최종: Neo4j에서 읽은 profile"은 profile 내용이 100% 같고(67개 카테고리 전부 일치), 결과도 같다.
  Neo4j에서 읽어도 결과가 그대로 재현된다는 확인이다.
- 그 두 행의 LLM 호출 17건은 모두 이전 실행의 캐시 적중이라 **이번에 새로 받은 Gemini 응답이 아니다.**
  NIST 1번·r1·r2 실행은 프롬프트가 바뀌어 실제로 호출했다(각각 11건, 9건, 5건이고 r2는 캐시 적중이 3건 더 있었다).

## 3. 해석할 때 주의

| 수치 | 성격 |
|---|---|
| 77.3% (deploy profile) | profile을 만들 때 22건을 쓰지 않았다. 다만 profile은 한국 인증기준 248개로 만들고 다듬은 것이라 NIST와는 도메인이 다르다. |
| 86.4% (NIST 1번 방식) | 22건의 문장·정답·NIST ID를 profile에 넣지 않았고 틀린 항목을 보며 반복 수정하지도 않았다. **그러나** 'NIST 용어를 67개 중 어느 카테고리에 붙일지' 작성자가 정했고, 그 판단을 정답지와 1차 결과를 이미 본 상태에서 했다. 정답지 영향이 섞였을 수 있다. |
| 86.4% → 90.9% (3번 방식) | 채점하는 22건의 **오답을 보고 profile을 고쳤다.** '들어맞게 맞춘 값'이며 일반화 성능이 아니다. |

- 22건은 표본이 작다. 1건 = 4.5%p.
- 3번 방식 r1에서 한 카테고리(BCP-01)에 어휘를 붙이자 인접 항목(RC.CO-03)이 끌려가는 부작용이 실제로 났다.
  그래서 r2에서 되돌렸다. 이런 부작용은 22건 안에서 보인 것이고, 보이지 않는 항목에서는 더 클 수 있다.
- 일반화 성능을 보려면 profile 작성에 쓰지 않은 NIST 항목이 더 필요하다.
- **정답지 재검토 후보:** RC.RP-01(정답 BCP-01)은 의미가 '사고 후 복구 계획 실행'에 가까워서 정답지 자체가 애매할 수 있다.
  어떤 profile에서도 정답이 임베딩 5~9위였다.
- 선배님 저장소의 93.55%는 1,569개 설정을 돌려 가장 잘 나온 하나를 고른 값이다(선배님 문서에도 명시). 이 저장소의 수치와
  직접 비교하지 않는다.

정답지(`data/nist_ground_truth_22.csv`) 출처: **Claude와 범서가 협의해 작성하고 Gemini로 교차검증했다.**

## 4. 파일 구성

### 스크립트 (`algorithm/`)

| 파일 | 설명 |
|---|---|
| `nist_hybrid_prototype.py` | 가설 B. 카테고리 텍스트는 이름만(`도메인명 카테고리명`)으로 두고 임베딩만 dense → hybrid로 바꿔 효과를 분리해서 본다. Neo4j 읽기 전용. dense 순위도 같이 기록한다(Phase D 재현 확인용). |
| `nist_full_pipeline.py` | 통합(CSV). deploy profile CSV + `classify.classify()` 로 hybrid + LLM 재판정까지. 가설 A 대조군(dense, hybrid 단독)도 같이 잰다. |
| `nist_llm_hybrid_prototype.py` | **최종.** `nist_full_pipeline.py` 와 같은 로직인데 카테고리 profile을 Neo4j(`Domain-[:HAS_CATEGORY]->Category`)에서 직접 읽는다. `--profile-prefix` 로 속성 세트(생략 = deploy, `nist_`, `nist_r1_`, `nist_r2_`)를 고른다. 프롬프트 유출 검사(NIST ID, 정답/점수/순위 표현, 후보 순서가 임베딩 순위와 같은지)를 통과해야 결과를 저장하고, 결과 파일이 있으면 덮어쓰지 않는다. `--dry-run` 은 가짜 LLM으로 코드 경로만 확인한다. |
| `build_nist_profiles.py` | 1번 방식. deploy profile 뒤에 NIST CSF 2.0 용어의 영어 설명 한 문장과 키워드를 덧붙여 `data/category_profiles_nist.csv` 를 만든다. |
| `build_nist_profiles_tuned.py` | 3번 방식. 라운드별 변경분(`DELTAS`)을 이전 라운드 파일에 적용해 `category_profiles_nist_tuned_r{N}.csv` 를 새로 만든다. 모든 라운드의 변경 내용이 코드에 남아 있다. |
| `load_taxonomy_to_neo4j.py` | deploy profile CSV로 Domain 16 / Category 67 / HAS_CATEGORY 67을 적재(MERGE만, 삭제 없음). |
| `load_profile_to_neo4j.py` | profile CSV를 Category의 접두사 붙은 속성으로 더한다(기존 속성 보존). |
| `summarize_results.py` | 결과 CSV에서 위 비교표를 계산해 `results/summary_table.csv` 로 저장한다. |
| `sbert_prototype.py` | 원본(선배님 저장소) 그대로. `nist_hybrid_prototype.py` 가 Neo4j 조회(`fetch_categories`)에 쓴다. |
| `benchmark_lofo.py`, `pure/scoring.py`, `pure/classify_config.json`, `llm/llm_client.py` | `classify.py` 의 의존 파일. 원본 그대로. 마진 임계값(0.202556)과 K=15는 `classify_config.json` 에 있다. |
| `../classify.py` | 배포용 분류기. 원본에서 **내용 수정 없이** 복사하고 상단에 출처 주석 세 줄만 추가했다. |

### 데이터 (`data/`)

| 파일 | 설명 |
|---|---|
| `nist_ground_truth_22.csv` | 정답지. 입력으로는 `text_en` 한 문장만 쓰고 NIST ID·정답은 LLM 프롬프트에 넣지 않는다. |
| `category_profiles_deploy.csv` | 한국 인증기준 248개 항목으로 만든 67개 Category 설명서(원본). |
| `category_profiles_nist.csv` | 1번 방식 profile. |
| `category_profiles_nist_tuned_r1.csv`, `..._r2.csv` | 3번 방식 라운드 1, 2. |

### 결과 (`algorithm/results/`)

`hypothesis_b/`(Phase D·가설 B), `full_pipeline/`(가설 A·통합), `neo4j_pipeline/`(최종, deploy),
`neo4j_pipeline_nist/`, `neo4j_pipeline_nist_r1/`, `neo4j_pipeline_nist_r2/` 각각에 항목별 결과 CSV와, Neo4j에서
읽어 온 profile 덤프(`profiles_from_neo4j.csv`)가 있다. `summary_table.csv` 는 비교표다.

진행 순서와 라운드별 변경 이유는 [`NIST_TUNING_LOG.md`](NIST_TUNING_LOG.md) 에 있다.

## 5. 실행 방법

사전 준비: Python 3.12, Neo4j 5.x, `requirements-lock.txt` 설치(`FlagEmbedding`, `sentence-transformers`, `neo4j` 등),
`BAAI/bge-m3` 모델 다운로드(처음 한 번).

```bash
# 0) 설정: algorithm/.env.example 을 algorithm/.env 로 복사해 Neo4j 정보를 넣는다.
#    LLM 재판정에는 같은 .env 에 LLM_PROVIDER=gemini, GOOGLE_API_KEY=..., LLM_MODEL=... 도 필요하다 (llm/llm_client.py 참고).
#    .env 는 .gitignore 에 들어 있어 올라가지 않는다.

# 1) Neo4j 적재
python algorithm/load_taxonomy_to_neo4j.py
python algorithm/load_profile_to_neo4j.py --csv data/category_profiles_nist.csv           --prefix nist_
python algorithm/load_profile_to_neo4j.py --csv data/category_profiles_nist_tuned_r1.csv  --prefix nist_r1_
python algorithm/load_profile_to_neo4j.py --csv data/category_profiles_nist_tuned_r2.csv  --prefix nist_r2_

# 2) 가짜 LLM으로 코드 경로와 프롬프트 유출 검사부터 (결과는 *_dryrun 폴더, 수치는 의미 없음)
python algorithm/nist_llm_hybrid_prototype.py --input data/nist_ground_truth_22.csv --dry-run

# 3) 실제 실행 (--profile-prefix 를 생략하면 deploy profile)
python algorithm/nist_llm_hybrid_prototype.py --input data/nist_ground_truth_22.csv
python algorithm/nist_llm_hybrid_prototype.py --input data/nist_ground_truth_22.csv --profile-prefix nist_r2_

# 4) 비교표
python algorithm/summarize_results.py
```

이미 결과 폴더가 있으면 스크립트는 덮어쓰지 않고 멈춘다(이전 수치를 나중에 대조하기 위해서). 다시 돌리려면 폴더 이름을 바꾸거나
옮긴다. LLM 응답은 프롬프트 해시로 캐시되므로 같은 프롬프트를 다시 돌리면 API를 부르지 않는다.

## 6. 올리지 않은 것

- `algorithm/.env` (접속 정보·API 키)
- 카테고리 임베딩 캐시 `deploy_cache/` (폴더당 40~100MB, 처음 실행하면 다시 생성됨)
- 가짜 LLM(`--dry-run`) 실행 결과 `*_dryrun/` (코드 경로 검증용이라 수치가 의미 없음)
- LLM 응답 캐시 `algorithm/results/llm/` (다시 실행하면 Gemini를 새로 호출한다)
- 이전 세션의 탐색용 스크립트와 결과(`run_nist_csf_test.py`, `simple_hybrid_mapper.py`, `llm_postprocess.py`, `results/nist_csf/` 등)
