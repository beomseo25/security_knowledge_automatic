"""NIST 22건 결과 CSV들에서 조건별 Top-1 / Top-3 를 계산해 비교표를 만든다.

모든 수치는 algorithm/results/ 아래 결과 CSV에서 직접 센다. 손으로 옮겨 적은 값이 없다.

- Top-3 (임베딩 단독): 각 결과 CSV의 *_top3_match 열.
- Top-3 (hybrid+LLM 최종): LLM(또는 자동확정)이 정한 최종 답을 1위로 두고, 2~3위는 임베딩 순위에서
  최종 답과 겹치지 않는 후보를 차례로 채운다. LLM은 하나만 고르므로 이 정의로만 Top-3 를 말할 수 있다.
  (임베딩 순위는 emb_top1~3 열까지만 저장되어 있다.)

사용법 (저장소 루트에서):
    python algorithm/summarize_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT_CSV = RESULTS / "summary_table.csv"


def read(rel: str) -> list[dict]:
    with open(RESULTS / rel, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def count(rows: list[dict], col: str) -> int:
    return sum(r[col] == "True" for r in rows)


def final_top3_hit(r: dict) -> bool:
    final = r["final_category_id"]
    rest = [c for c in (r["emb_top1"], r["emb_top2"], r["emb_top3"]) if c != final]
    return r["answer_category_id"].strip() in ([final] + rest[:2])


def fmt(k: int, n: int) -> str:
    return "%d/%d (%.1f%%)" % (k, n, 100 * k / n)


def main() -> int:
    table: list[tuple[str, str, str, str, str]] = []   # (조건, profile, 방식, Top-1, Top-3)

    hb = read("hypothesis_b/nist_hypothesis_b_hybrid_en.csv")
    n = len(hb)
    table.append(("Phase D", "이름만", "dense", fmt(count(hb, "dense_top1_match"), n), fmt(count(hb, "dense_top3_match"), n)))
    table.append(("가설 B", "이름만", "hybrid", fmt(count(hb, "hybrid_top1_match"), n), fmt(count(hb, "hybrid_top3_match"), n)))

    fp = read("full_pipeline/nist_full_pipeline.csv")
    n = len(fp)
    table.append(("가설 A", "deploy CSV (한국 인증기준용)", "dense", fmt(count(fp, "rich_dense_top1_match"), n), fmt(count(fp, "rich_dense_top3_match"), n)))
    table.append(("가설 A", "deploy CSV (한국 인증기준용)", "hybrid", fmt(count(fp, "emb_top1_match"), n), fmt(count(fp, "emb_top3_match"), n)))
    table.append(("통합 (CSV)", "deploy CSV (한국 인증기준용)", "hybrid + LLM", fmt(count(fp, "final_match"), n),
                  fmt(sum(final_top3_hit(r) for r in fp), n)))

    runs = [("neo4j_pipeline", "최종: Neo4j에서 읽은 deploy profile", "독립 값"),
            ("neo4j_pipeline_nist", "NIST 공개 용어 profile (1번 방식)", "정답지 영향 가능"),
            ("neo4j_pipeline_nist_r1", "NIST profile 튜닝 r1 (3번 방식)", "in-sample 튜닝"),
            ("neo4j_pipeline_nist_r2", "NIST profile 튜닝 r2 (3번 방식)", "in-sample 튜닝")]
    for folder, label, note in runs:
        rows = read("%s/nist_neo4j_pipeline.csv" % folder)
        n = len(rows)
        table.append((label, note, "dense", fmt(count(rows, "rich_dense_top1_match"), n), fmt(count(rows, "rich_dense_top3_match"), n)))
        table.append((label, note, "hybrid", fmt(count(rows, "emb_top1_match"), n), fmt(count(rows, "emb_top3_match"), n)))
        table.append((label, note, "hybrid + LLM", fmt(count(rows, "final_match"), n), fmt(sum(final_top3_hit(r) for r in rows), n)))

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "profile / note", "method", "top1", "top3"])
        w.writerows(table)

    print("| 조건 | profile / 비고 | 방식 | Top-1 | Top-3 |")
    print("|---|---|---|---|---|")
    for row in table:
        print("| " + " | ".join(row) + " |")
    print("\n저장: %s" % OUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
