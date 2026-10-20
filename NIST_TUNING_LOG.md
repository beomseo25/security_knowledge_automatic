# NIST CSF 2.0 22건 — profile 방식별 진행 기록

작성일 2026-10-20. 입력은 `data/nist_ground_truth_22.csv` 의 `text_en` 한 문장뿐이고, 카테고리 profile은
모두 Neo4j(`Domain-[:HAS_CATEGORY]->Category`)에서 읽는다. 파이프라인은 `classify.classify()`
(하이브리드 임베딩 + 마진 하위 구간 LLM 재판정, 임계값 margin<0.202556, K=15)를 수정 없이 쓴다.
LLM은 gemini-3.5-flash-lite. 모든 실행에서 프롬프트 유출 검사 위반 0건.

## 한눈에 보기

| 시도 | profile | 채점 방식 | 임베딩 hybrid Top-1 | 최종 Top-1 (hybrid+LLM) | 비고 |
|---|---|---|---|---|---|
| Phase D | 이름만 | 22건 | 27.3% (dense) | - | 기준선 |
| deploy profile | `category_profiles_deploy.csv` (한국 인증기준용) | 22건, 독립 | 50.0% | 77.3% (17/22) | Neo4j에서 읽어도 CSV와 동일 |
| **1번: NIST 공개 용어** (`nist_`) | `category_profiles_nist.csv` | 22건, 정답지 영향 가능 | 77.3% | **86.4% (19/22)** | 틀린 항목을 보며 반복 수정하지 않음 |
| **3번 r1** (`nist_r1_`) | `category_profiles_nist_tuned_r1.csv` | 22건, **in-sample 튜닝** | 77.3% | 86.4% (19/22) | 2건 고치고 2건 새로 틀림 |
| **3번 r2** (`nist_r2_`) | `category_profiles_nist_tuned_r2.csv` | 22건, **in-sample 튜닝** | 86.4% | **90.9% (20/22)** | r1의 부작용 되돌림 + 보강 |

주의: 1번, 3번 수치는 일반화 성능이 아니다. 특히 3번은 채점하는 22건의 오답을 보고 profile을 고쳤으므로
'들어맞게 맞춘 값'이다. 선배님 deploy profile의 77.3%만 profile 작성에 22건을 쓰지 않은 값이다.

## 진행 순서

### 0. Neo4j 준비
- 새 인스턴스의 기존 그래프(노드 335, 관계 331)를 JSON으로 백업한 뒤 초기화 (사용자 승인).
- `category_profiles_deploy.csv` 로 Domain 16, Category 67, HAS_CATEGORY 67 적재.
- `algorithm/nist_llm_hybrid_prototype.py` : profile을 Neo4j에서 읽는 파이프라인 (`--profile-prefix` 로 속성 세트 선택).

### 1. deploy profile (독립 값)
- 결과 17/22 = 77.3%. 임베딩 hybrid 50.0%, LLM이 오답 6건을 정답으로 고침(정답→오답 0).
- 결과 파일: `algorithm/results/neo4j_pipeline/nist_neo4j_pipeline.csv`
- LLM 호출 17건은 전부 이전 실행의 캐시 적중이었다 (새 API 호출 0).

### 2. 1번 방식 — NIST 공개 용어로 profile 작성 (`nist_`)
- `algorithm/build_nist_profiles.py` 가 deploy profile 뒤에 NIST CSF 2.0 용어의 영어 설명 한 문장과
  영어 키워드를 덧붙인다. 22건의 문장·정답·NIST ID는 넣지 않았다.
- 프롬프트 유출 검사가 `correct` 문자열(`corrective action`, `correction`)을 오탐으로 잡아 한 번 중단 →
  검사를 느슨하게 하지 않고 profile 문구를 `remedial action`, `rectification` 으로 바꿨다. 점수는 보기 전.
- 결과 19/22 = 86.4%. 임베딩 hybrid 77.3%, LLM은 3건 고치고 1건(GV.OV-01) 깨뜨림.
- 한계: 작성자가 이미 정답지와 1차 결과를 본 상태에서 'NIST 용어를 어느 카테고리에 붙일지' 정했다.
- 결과 파일: `algorithm/results/neo4j_pipeline_nist/nist_neo4j_pipeline.csv`

### 3. 3번 방식 — 오답을 보고 다듬기 (`build_nist_profiles_tuned.py`, 변경분은 DELTAS에 있음)

**r1** (오답 3건: GV.OV-01, GV.SC-01, RC.RP-01)
- GOV-03 에 '전략 결과 검토·조정' 어휘 보강, RSK-01 에서 `risk management strategy` 키워드 삭제.
- SUP-01 의 공급망 전략 어휘를 SUP-02 로 이동 (v1에서 잘못 붙인 것).
- BCP-01 에 '복구 실행' 어휘 추가 (RC.RP-01 의 정답이 BCP-01 이라서. 가장 억지스러운 수정).
- 결과 19/22: GV.OV-01, GV.SC-01 고침. 그러나 RC.CO-03 이 BCP-01 로 끌려감(위 BCP-01 수정의 부작용),
  RS.MA-01 은 LLM이 INC-01 로 바꿈. RC.RP-01 은 여전히 오답(정답 순위 9위→5위).
- 결과 파일: `algorithm/results/neo4j_pipeline_nist_r1/nist_neo4j_pipeline.csv`

**r2** (오답 3건: RS.MA-01, RC.RP-01, RC.CO-03)
- BCP-01 변경을 r0 상태로 되돌림 (RC.RP-01 은 맞추지 않기로 함).
- INC-02 에 '복구 활동 소통' 어휘, INC-03 에 '사고 선언 후 대응 계획 실행' 어휘 보강.
- 결과 20/22 = 90.9%. 임베딩 hybrid 86.4%. 남은 오답: RS.AN-03 (임베딩은 정답이었으나 LLM이 INC-04 로 바꿈),
  RC.RP-01 (정답 BCP-01 이 임베딩 9위).
- 결과 파일: `algorithm/results/neo4j_pipeline_nist_r2/nist_neo4j_pipeline.csv`

## 알게 된 것
- profile 어휘가 가장 크게 움직인 단계는 임베딩이다. deploy 50.0% → NIST 용어 77.3% → r2 86.4%.
- LLM 재판정이 더한 정답 수(순효과)는 deploy +6, 1번 방식 +2, r1 +2, r2 +1 로 작아졌다. 임베딩이 좋아질수록 LLM이 정답을 뒤집는 위험이 보인다
  (GV.OV-01, RS.AN-03).
- 한 카테고리에 어휘를 붙이면 인접 카테고리 항목이 끌려가는 부작용이 실제로 났다 (r1 의 RC.CO-03).
- RC.RP-01 은 정답이 BCP-01 인데 의미가 '사고 후 복구 계획 실행'에 가깝다. 정답지 자체가 애매할 수 있어
  재검토 대상이다.
- 22건은 표본이 작다 (1건 = 4.5%p). 일반화 성능을 보려면 profile 작성에 쓰지 않은 NIST 항목이 더 필요하다.

## 재현 명령
```
python algorithm/build_nist_profiles.py                                   # r0 (1번 방식) profile
python algorithm/build_nist_profiles_tuned.py --round 1                   # r1
python algorithm/build_nist_profiles_tuned.py --round 2                   # r2
NEO4J_PASSWORD=... python algorithm/load_profile_to_neo4j.py --csv data/category_profiles_nist_tuned_r2.csv --prefix nist_r2_
NEO4J_PASSWORD=... python algorithm/nist_llm_hybrid_prototype.py --input data/nist_ground_truth_22.csv --profile-prefix nist_r2_ [--dry-run]
```
