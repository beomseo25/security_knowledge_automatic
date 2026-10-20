"""가설 B 단독 검증: 카테고리 텍스트는 고정하고 임베딩 방식만 dense → hybrid로 바꾼다.

- 카테고리 텍스트: sbert_prototype.py와 동일 (f"{domain_name} {category_name}",
  Neo4j에서 sbert_prototype.fetch_categories로 조회). category_profiles.csv는 쓰지 않는다.
- 유사도: security-knowledge-graph-neo4j/algorithm/benchmark_bge_m3_hybrid.py의
  encode_bge_m3(195~210행), compute_score_matrices(213~235행)를 수정 없이 복사.
  (같은 이름의 -Automatic 사본은 다른 구현이라 쓰지 않는다.)
- 점수: 같은 파일 WEIGHTS(81~83행)의 detail 3개 값만 사용. meta/final 이중구조는 쓰지 않는다.
      score = 0.6 * dense + 0.1 * sparse + 0.3 * colbert
- 대조군: 같은 실행에서 dense 단독 순위도 함께 기록한다. 이 값이 Phase D(27.3%/68.2%)를
  재현하지 못하면 환경 차이(FlagEmbedding fp16 vs SentenceTransformer 등)가 섞여 있다는 뜻이다.

읽기 전용(Read-Only): Neo4j를 변경하지 않는다. 기존 결과 파일을 덮어쓰지 않는다.

사용법 (저장소 루트에서, venv 활성화 후):

    python algorithm/nist_hybrid_prototype.py
    python algorithm/nist_hybrid_prototype.py --input data/nist_ground_truth_22.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from sbert_prototype import fetch_categories, load_config

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "nist_ground_truth_22.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "hypothesis_b" / "nist_hypothesis_b_hybrid.csv"

MODEL_NAME_BGE_M3 = "BAAI/bge-m3"

# security-knowledge-graph-neo4j/algorithm/benchmark_bge_m3_hybrid.py 81~83행 그대로.
W_DENSE = 0.6
W_SPARSE = 0.1
W_COLBERT = 0.3

FOCUS_IDS = ("RS.MA-01", "RS.AN-03", "RS.MI-01")


# ---------------------------------------------------------------------------
# security-knowledge-graph-neo4j/algorithm/benchmark_bge_m3_hybrid.py 에서 수정 없이 복사
# ---------------------------------------------------------------------------

def encode_bge_m3(model, sentences: list[str]) -> dict:
    return model.encode(
        sentences,
        batch_size=12,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )


def compute_score_matrices(model, left_output: dict, right_output: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_matrix = np.asarray(left_output["dense_vecs"]) @ np.asarray(right_output["dense_vecs"]).T

    n_left = len(left_output["lexical_weights"])
    n_right = len(right_output["lexical_weights"])
    sparse_matrix = np.zeros((n_left, n_right), dtype=float)
    colbert_matrix = np.zeros((n_left, n_right), dtype=float)

    for i in range(n_left):
        for j in range(n_right):
            sparse_matrix[i, j] = float(
                model.compute_lexical_matching_score(
                    left_output["lexical_weights"][i], right_output["lexical_weights"][j]
                )
            )
            colbert_matrix[i, j] = float(
                model.colbert_score(left_output["colbert_vecs"][i], right_output["colbert_vecs"][j])
            )

    return dense_matrix, sparse_matrix, colbert_matrix


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path, text_column: str) -> list[dict]:
    if not path.exists():
        print(f"[오류] 입력 파일이 없습니다: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    missing = {text_column, "answer_category_id"} - set(rows[0] if rows else {})
    if missing:
        print(f"[오류] 입력 CSV에 필요한 열이 없습니다: {sorted(missing)}")
        sys.exit(1)
    return rows


def rank_of(ranked_ids: list[str], answer: str) -> int | None:
    return ranked_ids.index(answer) + 1 if answer in ranked_ids else None


def evaluate(rows: list[dict], categories, dense, sparse, colbert) -> list[dict]:
    hybrid = W_DENSE * dense + W_SPARSE * sparse + W_COLBERT * colbert
    cat_ids = [c.category_id for c in categories]

    results = []
    for i, row in enumerate(rows):
        answer = row["answer_category_id"].strip()
        out = dict(row)
        for label, matrix in (("hybrid", hybrid), ("dense", dense)):
            order = np.argsort(-matrix[i])
            ranked = [cat_ids[k] for k in order]
            for r in range(3):
                out[f"{label}_top{r + 1}"] = ranked[r]
                out[f"{label}_top{r + 1}_score"] = round(float(matrix[i, order[r]]), 4)
            out[f"{label}_top1_match"] = ranked[0] == answer
            out[f"{label}_top3_match"] = answer in ranked[:3]
            out[f"{label}_answer_rank"] = rank_of(ranked, answer)
        if answer in cat_ids:
            j = cat_ids.index(answer)
            out["answer_dense"] = round(float(dense[i, j]), 4)
            out["answer_sparse"] = round(float(sparse[i, j]), 4)
            out["answer_colbert"] = round(float(colbert[i, j]), 4)
        results.append(out)
    return results


def print_report(results: list[dict]) -> None:
    n = len(results)
    print(f"\n=== 결과 (n={n}) ===")
    for label in ("dense", "hybrid"):
        t1 = sum(r[f"{label}_top1_match"] for r in results)
        t3 = sum(r[f"{label}_top3_match"] for r in results)
        print(f"{label:>6}: Top-1 {t1}/{n} ({t1 / n:.1%})  Top-3 {t3}/{n} ({t3 / n:.1%})")
    print("  (Phase D 기록: Top-1 27.3%, Top-3 68.2%)")

    print("\n=== 집중 항목 ===")
    for focus in FOCUS_IDS:
        hit = [r for r in results if any(str(v).strip() == focus for v in r.values())]
        if not hit:
            print(f"{focus}: 입력 CSV에서 찾지 못함")
            continue
        r = hit[0]
        print(
            f"{focus}: 정답 {r['answer_category_id']} | "
            f"dense top3 {r['dense_top1']},{r['dense_top2']},{r['dense_top3']} (정답 {r['dense_answer_rank']}위) | "
            f"hybrid top3 {r['hybrid_top1']},{r['hybrid_top2']},{r['hybrid_top3']} (정답 {r['hybrid_answer_rank']}위)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="가설 B 단독 검증 (카테고리 텍스트 고정, hybrid 유사도)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-column", default="text_ko", help="Control 문장으로 쓸 입력 CSV 열 (기본: text_ko)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"[안내] {args.output} 가 이미 있습니다. 덮어쓰지 않으려면 --output 으로 새 경로를 지정하세요.")
        return 1

    rows = load_ground_truth(args.input, args.text_column)
    categories = fetch_categories(load_config())
    category_texts = [f"{c.domain_name} {c.category_name}" for c in categories]
    print(f"입력 {len(rows)}개 (열: {args.text_column}), Category {len(categories)}개")

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(MODEL_NAME_BGE_M3, use_fp16=True)

    control_output = encode_bge_m3(model, [r[args.text_column] for r in rows])
    category_output = encode_bge_m3(model, category_texts)
    dense, sparse, colbert = compute_score_matrices(model, control_output, category_output)

    results = evaluate(rows, categories, dense, sparse, colbert)
    print_report(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
