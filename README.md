# NIST CSF 2.0 → 보안 분류체계 매핑 검증

## 문제

NIST CSF 2.0 서브카테고리 22개(영문 한 문장)를 우리 보안 분류체계 67개 Category 중 하나에 자동으로 배정하고 싶다.
기존 방식(BGE-M3 하이브리드 임베딩으로 후보를 정렬하고, 1·2위 점수 차이가 작은 항목만 LLM이 다시 판정)은 한국 인증기준
248개 항목에서 만든 것이라, NIST 문장에서도 되는지 확인했다.

정답지(`data/nist_ground_truth_22.csv`)는 Claude와 범서가 협의해 만들고 Gemini로 교차검증했다.

## 시도와 결과 (n=22)

| # | 시도 | Category 설명(profile) | 방식 | Top-1 | Top-3 |
|---|---|---|---|---|---|
| 1 | Phase D | 이름만 | dense | 27.3% (6/22) | 68.2% |
| 2 | 가설 B | 이름만 | hybrid | 36.4% (8/22) | 68.2% |
| 3 | 가설 A | 한국 인증기준용 profile | dense | 45.5% (10/22) | 81.8% |
| 4 | 가설 A | 한국 인증기준용 profile | hybrid | 50.0% (11/22) | 68.2% |
| 5 | 통합 (Neo4j에서 읽어도 동일) | 한국 인증기준용 profile | hybrid + LLM | **77.3% (17/22)** | 81.8% |
| 6 | NIST 용어 profile | NIST CSF 2.0 용어 추가 | hybrid + LLM | 86.4% (19/22) | 95.5% |
| 7 | profile 튜닝 r1 | 오답을 보고 수정 | hybrid + LLM | 86.4% (19/22) | 95.5% |
| 8 | profile 튜닝 r2 | r1 부작용 되돌리고 보강 | hybrid + LLM | 90.9% (20/22) | 95.5% |

- **원인:** NIST에서 점수가 낮았던 이유는 방식이 아니라 Category 설명이 한국 인증기준 어휘로만 되어 있어서였다.
  이름만 쓰면 27~36%, 설명(profile)을 쓰면 임베딩만으로 50%, LLM 재판정까지 77.3%.
- **6~8번은 일반화 성능이 아니다.** 6번은 정답지를 본 상태에서 profile을 썼고, 7·8번은 22건의 오답을 보고 profile을 고쳤다.
  profile 작성에 22건을 쓰지 않은 값은 5번의 77.3%뿐이다. 22건이라 1건이 4.5%p다.
- 남은 오답 2건(8번): RS.AN-03(임베딩은 맞혔으나 LLM이 바꿈), RC.RP-01(정답 BCP-01, 정답지 재검토 후보).
- hybrid + LLM 행의 Top-3는 LLM 최종 답을 1위로 두고 2~3위를 임베딩 순위로 채운 값이다.

시도별 변경 내용과 진행 순서는 [`NIST_TUNING_LOG.md`](NIST_TUNING_LOG.md)에 있다.

## 저장된 파일

```
classify.py                         배포용 분류기 (원본 그대로 + 출처 주석 3줄)
NIST_TUNING_LOG.md                  진행 기록
data/
  nist_ground_truth_22.csv          정답지
  category_profiles_deploy.csv      한국 인증기준용 profile (시도 3~5)
  category_profiles_nist.csv        NIST 용어 profile (시도 6)
  category_profiles_nist_tuned_r1.csv, _r2.csv   튜닝 profile (시도 7, 8)
algorithm/
  nist_hybrid_prototype.py          시도 1, 2 (이름만, dense vs hybrid)
  nist_full_pipeline.py             시도 3~5 (profile을 CSV에서 읽음)
  nist_llm_hybrid_prototype.py      시도 5~8 (profile을 Neo4j에서 읽음, 최종)
  build_nist_profiles.py            시도 6 profile 생성
  build_nist_profiles_tuned.py      시도 7, 8 profile 생성 (라운드별 변경분 포함)
  load_taxonomy_to_neo4j.py, load_profile_to_neo4j.py   Neo4j 적재
  summarize_results.py              위 표를 결과 CSV에서 계산
  sbert_prototype.py, benchmark_lofo.py, pure/, llm/    원본 그대로 (의존 파일)
  results/                          시도별 결과 CSV (hypothesis_b, full_pipeline, neo4j_pipeline*)
```
