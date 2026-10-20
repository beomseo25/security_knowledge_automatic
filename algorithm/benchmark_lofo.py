"""Leave-One-Framework-Out 정직 평가 하네스.

선배님 저장소의 `benchmark_bge_m3_hybrid.py`와 **같은 채점 방식**(문장 3종 x
Category 문장 6종 x Dense/Sparse/ColBERT, 행 단위 min-max 정규화, 1,569개 config)을
우리 데이터에 옮겨 놓되, 시도 9에서 확립한 정직한 규칙을 코드로 강제한다.

정직한 규칙 (시도 9)
--------------------
1. 홀드아웃 프레임워크를 **인코딩 단계에서부터** 봉인한다. 봉인 모드에서는
   홀드아웃 항목이 인코딩·점수·순위·출력 어디에도 등장하지 않는다.
2. Category 설명서는 train 프레임워크만 보고 쓴다 (이 스크립트 밖의 약속).
3. **config(가중치) 선택도 train 성적으로만 한다.** 1,569개 후보 전부를 train
   Top-1으로 줄 세워 1등을 뽑는다. 홀드아웃 성적은 선택에 절대 개입하지 않는다.
4. 그 config를 홀드아웃에 **딱 한 번** 적용해 채점한다 (`--reveal-test`).

`--reveal-test` 없이 실행하면 train 성적만 나온다. 설명서를 쓰거나 고치는 동안은
반드시 이 봉인 모드로만 돌린다.

읽기 전용: Neo4j를 읽지도 쓰지도 않는다 (CSV만 사용).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "lofo"
CONTROLS_CSV = DATA_DIR / "controls_from_cypher.csv"

MODEL_NAME = "BAAI/bge-m3"
FRAMEWORKS = ("CASP", "ISMS-P", "ISO27001")


class DataError(Exception):
    pass


# ---------------------------------------------------------------------------
# 문장 구성 (선배님 하네스와 동일한 형태)
# ---------------------------------------------------------------------------

def labeled_text(parts: list[tuple[str, str]]) -> str:
    return ". ".join(f"{label}: {value.strip()}" for label, value in parts if value.strip())


def build_control_text_groups(controls: list[dict]) -> dict[str, list[str]]:
    detail = [(c.get("original_detail") or c.get("title") or "").strip() for c in controls]

    compact: list[str] = []
    structured: list[str] = []
    for c in controls:
        seen: list[str] = []
        for key in ("original_level1", "original_level2", "title", "original_detail"):
            value = (c.get(key) or "").strip()
            if value and value not in seen:
                seen.append(value)
        compact.append(" ".join(seen))
        structured.append(labeled_text([
            ("상위 통제 분야", c.get("original_level1") or ""),
            ("세부 통제 항목", c.get("original_level2") or ""),
            ("통제 제목", c.get("title") or ""),
            ("통제 요구사항", c.get("original_detail") or ""),
        ]))

    groups = {"detail": detail, "metadata_compact": compact, "metadata_structured": structured}
    for name, texts in groups.items():
        if any(not t for t in texts):
            raise DataError(f"빈 Control 문장이 있습니다: {name}")
    return groups


def build_category_text_groups(profiles: list[dict]) -> dict[str, list[str]]:
    def field(p: dict, key: str) -> str:
        return (p.get(key) or "").strip()

    return {
        "label": [f"{field(p, 'domain_name')} {field(p, 'category_name')}" for p in profiles],
        "description": [labeled_text([
            ("보안 영역", field(p, "domain_name")),
            ("보안 Category", field(p, "category_name")),
            ("설명", field(p, "domain_description")),
        ]) for p in profiles],
        "profile_definition": [labeled_text([
            ("보안 영역", field(p, "domain_name")),
            ("보안 Category", field(p, "category_name")),
            ("상세 정의", field(p, "detail_description")),
        ]) for p in profiles],
        "profile_keywords": [labeled_text([
            ("보안 영역", field(p, "domain_name")),
            ("보안 Category", field(p, "category_name")),
            ("대표 키워드", field(p, "keywords")),
        ]) for p in profiles],
        "profile_full": [labeled_text([
            ("보안 영역", field(p, "domain_name")),
            ("보안 Category", field(p, "category_name")),
            ("상세 정의", field(p, "detail_description")),
            ("대표 키워드", field(p, "keywords")),
        ]) for p in profiles],
        "profile_exclusions": [labeled_text([
            ("이 Category에서 제외할 의미", field(p, "exclusions")),
        ]) for p in profiles],
    }


# ---------------------------------------------------------------------------
# 점수 계산
# ---------------------------------------------------------------------------

def minmax_rows(matrix: np.ndarray) -> np.ndarray:
    """항목 한 개(행) 안에서 Category 점수를 0~1로 편다."""
    minimum = matrix.min(axis=1, keepdims=True)
    scale = matrix.max(axis=1, keepdims=True) - minimum
    scale = np.where(scale == 0, 1.0, scale)
    return (matrix - minimum) / scale


def colbert_matrix(query_vectors, passage_vectors) -> np.ndarray:
    scores = np.empty((len(query_vectors), len(passage_vectors)), dtype=np.float32)
    for i, query in enumerate(query_vectors):
        for j, passage in enumerate(passage_vectors):
            token_scores = np.einsum("id,jd->ij", query, passage, optimize=False)
            scores[i, j] = float(token_scores.max(axis=1).mean())
    if not np.isfinite(scores).all():
        raise DataError("ColBERT 점수에 NaN/무한대가 있습니다.")
    return scores


def copy_model_output(output: dict) -> dict:
    """FlagEmbedding이 다음 encode 호출에서 재사용하는 버퍼에서 떼어낸다."""
    return {
        "dense_vecs": np.array(output["dense_vecs"], dtype=np.float32, copy=True),
        "lexical_weights": [dict(w) for w in output["lexical_weights"]],
        "colbert_vecs": [np.array(v, dtype=np.float32, copy=True) for v in output["colbert_vecs"]],
    }


def build_all_config_matrices(model, control_text_groups, category_text_groups, domain_index):
    """선배님 하네스와 같은 1,569개 config의 점수 행렬을 만든다.

    반환: {config_name: (n_controls, n_categories) 점수 행렬}
    """
    control_outputs: dict[str, dict] = {}
    for name, texts in control_text_groups.items():
        print(f"  Control 인코딩: {name} ({len(texts)}개)")
        control_outputs[name] = copy_model_output(model.encode(
            texts, batch_size=8, max_length=256,
            return_dense=True, return_sparse=True, return_colbert_vecs=True))

    matrices: dict[str, np.ndarray] = {}

    for cat_name, cat_texts in category_text_groups.items():
        print(f"  Category 인코딩: {cat_name} ({len(cat_texts)}개)")
        cat_out = copy_model_output(model.encode(
            cat_texts, batch_size=8, max_length=256,
            return_dense=True, return_sparse=True, return_colbert_vecs=True))

        for ctl_name, ctl_out in control_outputs.items():
            prefix = "" if ctl_name == "detail" else f"{ctl_name}__"
            dense = np.einsum("id,jd->ij", ctl_out["dense_vecs"], cat_out["dense_vecs"],
                              optimize=False).astype(np.float32)
            sparse = model.compute_lexical_matching_score(
                ctl_out["lexical_weights"], cat_out["lexical_weights"]).astype(np.float32)
            print(f"  ColBERT 채점: {ctl_name} x {cat_name}")
            colbert = colbert_matrix(ctl_out["colbert_vecs"], cat_out["colbert_vecs"])

            for mode, m in (("dense", dense), ("sparse", sparse), ("colbert", colbert)):
                matrices[f"{prefix}{cat_name}__{mode}"] = m

            official = 0.4 * dense + 0.2 * sparse + 0.4 * colbert
            matrices[f"{prefix}{cat_name}__official_0.4_0.2_0.4"] = official

            normalized = (minmax_rows(dense), minmax_rows(sparse), minmax_rows(colbert))
            for d_step in range(11):
                for s_step in range(11 - d_step):
                    c_step = 10 - d_step - s_step
                    weights = (d_step / 10, s_step / 10, c_step / 10)
                    if sum(w > 0 for w in weights) < 2:
                        continue
                    combined = sum(w * m for w, m in zip(weights, normalized))
                    matrices[f"{prefix}{cat_name}__grid__"
                             f"d{weights[0]:.1f}_s{weights[1]:.1f}_c{weights[2]:.1f}"] = combined

    # 제외 기준 감점 방식
    positive_weights = (0.7, 0.1, 0.2)
    negative_options = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.0, 0.5), (0.7, 0.1, 0.2))
    for ctl_name in control_text_groups:
        prefix = "" if ctl_name == "detail" else f"{ctl_name}__"
        pos_modes = tuple(minmax_rows(matrices[f"{prefix}profile_full__{m}"])
                          for m in ("dense", "sparse", "colbert"))
        neg_modes = tuple(minmax_rows(matrices[f"{prefix}profile_exclusions__{m}"])
                          for m in ("dense", "sparse", "colbert"))
        positive = sum(w * m for w, m in zip(positive_weights, pos_modes))
        for neg_weights in negative_options:
            negative = sum(w * m for w, m in zip(neg_weights, neg_modes))
            key = "_".join(f"{w:.1f}" for w in neg_weights)
            for step in range(1, 11):
                penalty = step * 0.05
                matrices[f"{prefix}profile_contrastive__negative_{key}__"
                         f"penalty_{penalty:.2f}"] = positive - penalty * negative

    # 항목 문장 앙상블
    profile_weight_options = ((0.5, 0.1, 0.4), (0.6, 0.1, 0.3), (0.7, 0.1, 0.2))

    def combined_profile(ctl_name: str, weights) -> np.ndarray:
        prefix = "" if ctl_name == "detail" else f"{ctl_name}__"
        modes = tuple(minmax_rows(matrices[f"{prefix}profile_full__{m}"])
                      for m in ("dense", "sparse", "colbert"))
        return sum(w * m for w, m in zip(weights, modes))

    combined_profiles = {(c, w): combined_profile(c, w)
                         for c in control_text_groups for w in profile_weight_options}

    for context_name in ("metadata_compact", "metadata_structured"):
        for detail_weights in profile_weight_options:
            detail_scores = combined_profiles[("detail", detail_weights)]
            for context_weights in profile_weight_options:
                context_scores = combined_profiles[(context_name, context_weights)]
                for step in range(1, 10):
                    dw = step / 10
                    matrices[
                        f"control_ensemble__detail_{dw:.1f}__{context_name}__"
                        f"dw_{'_'.join(map(str, detail_weights))}__"
                        f"cw_{'_'.join(map(str, context_weights))}"
                    ] = dw * detail_scores + (1.0 - dw) * context_scores

    # 큰 분류 먼저 고르고 그 안에서 세부 분류
    n_domains = int(domain_index.max()) + 1

    def aggregate_domain(scores: np.ndarray, method: str) -> np.ndarray:
        result = np.empty((len(scores), n_domains), dtype=np.float32)
        for d in range(n_domains):
            selected = scores[:, domain_index == d]
            if method == "max":
                result[:, d] = selected.max(axis=1)
            elif method == "mean":
                result[:, d] = selected.mean(axis=1)
            elif method == "top2_mean":
                k = min(2, selected.shape[1])
                result[:, d] = np.sort(selected, axis=1)[:, -k:].mean(axis=1)
            else:
                raise ValueError(method)
        return result

    for cat_weights in profile_weight_options:
        cat_scores = combined_profiles[("detail", cat_weights)]
        for domain_ctl in control_text_groups:
            for dom_weights in profile_weight_options:
                source = combined_profiles[(domain_ctl, dom_weights)]
                for method in ("max", "mean", "top2_mean"):
                    domain_scores = aggregate_domain(source, method)
                    predicted = np.argmax(domain_scores, axis=1)
                    allowed = domain_index[np.newaxis, :] == predicted[:, np.newaxis]
                    matrices[
                        f"hierarchical__domain_{domain_ctl}__{method}__"
                        f"dw_{'_'.join(map(str, dom_weights))}__"
                        f"cw_{'_'.join(map(str, cat_weights))}"
                    ] = np.where(allowed, cat_scores, -1e9)

    return matrices


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------

def top1_matches(scores: np.ndarray, gold_index: np.ndarray) -> int:
    top1 = np.argsort(-scores, axis=1, kind="stable")[:, 0]
    return int((top1 == gold_index).sum())


def score_rows(scores: np.ndarray, controls: list[dict], profiles: list[dict],
               gold_index: np.ndarray) -> list[dict]:
    """한 config의 점수 행렬로 항목별 상세 결과를 만든다 (마진 포함)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    cat_ids = [p["category_id"] for p in profiles]
    cat_names = [p["category_name"] for p in profiles]
    domain_of = {p["category_id"]: p["domain_id"] for p in profiles}

    rows = []
    for i, control in enumerate(controls):
        ranked = order[i]
        top = ranked[:3]
        gold = control["actual_category_id"]
        top1_id = cat_ids[top[0]]
        s1 = float(scores[i, top[0]])
        s2 = float(scores[i, top[1]])
        rank = int(np.where(np.array(cat_ids)[ranked] == gold)[0][0]) + 1
        rows.append({
            "control_id": control["control_id"],
            "framework": control["framework"],
            "title": control["title"],
            "actual_category_id": gold,
            "actual_category_name": control.get("actual_category_name", ""),
            "actual_domain_id": domain_of.get(gold, ""),
            "top1_category_id": top1_id,
            "top1_category_name": cat_names[top[0]],
            "top1_score": round(s1, 6),
            "top2_category_id": cat_ids[top[1]],
            "top2_score": round(s2, 6),
            "top3_category_id": cat_ids[top[2]],
            "top3_score": round(float(scores[i, top[2]]), 6),
            "margin": round(s1 - s2, 6),
            "top1_match": top1_id == gold,
            "top3_match": gold in [cat_ids[t] for t in top],
            "top1_domain_match": domain_of.get(top1_id) == domain_of.get(gold),
            "actual_rank_among_67": rank,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "top1": 0, "top3": 0, "domain": 0}
    return {
        "n": n,
        "top1": sum(r["top1_match"] for r in rows),
        "top3": sum(r["top3_match"] for r in rows),
        "domain": sum(r["top1_domain_match"] for r in rows),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간. 표본이 작을 때 정규근사보다 정확하다."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def coverage_curve(rows: list[dict], steps: int = 41) -> list[dict]:
    """마진 기준 T를 올려가며 (자동 처리 비율, 그 구간 Top-1 정확도)를 낸다."""
    if not rows:
        return []
    margins = sorted(r["margin"] for r in rows)
    hi = margins[-1]
    out = []
    for i in range(steps):
        t = hi * i / (steps - 1) if steps > 1 else 0.0
        kept = [r for r in rows if r["margin"] >= t]
        if not kept:
            out.append({"threshold": round(t, 6), "n_auto": 0, "coverage": 0.0,
                        "top1_correct": 0, "top1_accuracy": ""})
            continue
        correct = sum(r["top1_match"] for r in kept)
        out.append({
            "threshold": round(t, 6),
            "n_auto": len(kept),
            "coverage": round(len(kept) / len(rows), 4),
            "top1_correct": correct,
            "top1_accuracy": round(correct / len(kept), 4),
        })
    return out


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------

def load_controls(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["control_id"])
    return rows


def load_profiles(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 67:
        raise DataError(f"Category Profile이 67개가 아니라 {len(rows)}개입니다: {path}")
    for r in rows:
        for key in ("detail_description", "keywords"):
            if not (r.get(key) or "").strip():
                raise DataError(f"{r['category_id']}의 {key}가 비어 있습니다.")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOFO 정직 평가 (train에서만 config 선택)")
    p.add_argument("--holdout", required=True, choices=FRAMEWORKS,
                   help="봉인할 프레임워크")
    p.add_argument("--profiles", type=Path, required=True, help="이 폴드의 Category 설명서 CSV")
    p.add_argument("--controls", type=Path, default=CONTROLS_CSV)
    p.add_argument("--reveal-test", action="store_true",
                   help="홀드아웃 봉인을 해제하고 최종 채점한다 (폴드당 딱 한 번)")
    p.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None, help="결과 파일 이름에 붙일 꼬리표")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag or args.holdout.lower()

    controls_all = load_controls(args.controls)
    profiles = load_profiles(args.profiles)

    train = [c for c in controls_all if c["framework"] != args.holdout]
    test = [c for c in controls_all if c["framework"] == args.holdout]
    if not test:
        raise DataError(f"홀드아웃 {args.holdout}에 해당하는 항목이 없습니다.")

    print("=" * 66)
    print(f"LOFO  홀드아웃={args.holdout}  train={len(train)}개  test={len(test)}개")
    print(f"설명서: {args.profiles}")
    print(f"모드  : {'봉인 해제 (최종 채점)' if args.reveal_test else '봉인 (train만)'}")
    print("=" * 66)

    # 봉인 모드에서는 홀드아웃 항목을 인코딩 자체에 넣지 않는다.
    controls = controls_all if args.reveal_test else train
    train_mask = np.array([c["framework"] != args.holdout for c in controls])

    cat_ids = [p["category_id"] for p in profiles]
    cat_pos = {cid: i for i, cid in enumerate(cat_ids)}
    gold_index = np.array([cat_pos[c["actual_category_id"]] for c in controls], dtype=int)

    domain_ids = []
    for p in profiles:
        if p["domain_id"] not in domain_ids:
            domain_ids.append(p["domain_id"])
    domain_index = np.array([domain_ids.index(p["domain_id"]) for p in profiles], dtype=int)

    control_texts = build_control_text_groups(controls)
    category_texts = build_category_text_groups(profiles)

    from FlagEmbedding import BGEM3FlagModel
    print(f"모델 로딩: {MODEL_NAME}")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=False)

    matrices = build_all_config_matrices(model, control_texts, category_texts, domain_index)
    print(f"\nconfig 후보 {len(matrices)}개 생성")

    # --- config 선택: TRAIN 행만 사용한다 ---
    train_gold = gold_index[train_mask]
    ranking = []
    for name, m in matrices.items():
        ranking.append((top1_matches(m[train_mask], train_gold), name))
    ranking.sort(key=lambda x: (-x[0], x[1]))
    best_train_hits, best_name = ranking[0]
    print(f"train 1등 config: {best_name}")
    print(f"  train Top-1 = {best_train_hits}/{int(train_mask.sum())} "
          f"({best_train_hits / int(train_mask.sum()) * 100:.2f}%)")

    best = matrices[best_name]
    train_rows = score_rows(best[train_mask], [c for c, k in zip(controls, train_mask) if k],
                            profiles, train_gold)
    train_sum = summarize(train_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "holdout_framework": args.holdout,
        "profiles_csv": str(args.profiles),
        "n_train": len(train),
        "n_test": len(test),
        "n_configs": len(matrices),
        "selected_config": best_name,
        "selected_on": "train only",
        "train": train_sum,
        "revealed": bool(args.reveal_test),
    }

    print()
    print(f"[TRAIN] {train_sum['n']}개  Top-1 {train_sum['top1']}/{train_sum['n']} "
          f"({train_sum['top1']/train_sum['n']:.1%})  "
          f"Top-3 {train_sum['top3']}/{train_sum['n']}  "
          f"Domain {train_sum['domain']}/{train_sum['n']}")

    if not args.reveal_test:
        print(f"[TEST]  홀드아웃 {len(test)}개 — 봉인 상태 (인코딩조차 하지 않음)")
        print("        최종 채점은 --reveal-test 로 딱 한 번.")
        # 봉인 검증: 어떤 출력 구조에도 홀드아웃 항목이 없어야 한다
        holdout_ids = {c["control_id"] for c in test}
        leaked = holdout_ids & {r["control_id"] for r in train_rows}
        if leaked:
            raise DataError(f"봉인 위반: 홀드아웃 항목이 결과에 있습니다: {sorted(leaked)[:5]}")
        assert len(controls) == len(train), "봉인 모드인데 홀드아웃이 인코딩에 포함됐습니다."
        print("        봉인 검증 통과 (결과·인코딩 어디에도 홀드아웃 항목 없음)")
        (args.outdir / f"{tag}_sealed_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=" * 66)
        return 0

    # --- 봉인 해제: 선택된 config를 홀드아웃에 딱 한 번 적용 ---
    test_mask = ~train_mask
    test_rows = score_rows(best[test_mask], [c for c, k in zip(controls, test_mask) if k],
                           profiles, gold_index[test_mask])
    test_sum = summarize(test_rows)
    lo, hi = wilson_interval(test_sum["top1"], test_sum["n"])
    result["test"] = test_sum
    result["test_top1_accuracy"] = test_sum["top1"] / test_sum["n"]
    result["test_top1_ci95"] = [lo, hi]

    print(f"[TEST]  홀드아웃 {test_sum['n']}개 — 봉인 해제 ★")
    print(f"  Top-1 : {test_sum['top1']}/{test_sum['n']} = {test_sum['top1']/test_sum['n']:.1%}"
          f"   (95% 신뢰구간 {lo:.1%} ~ {hi:.1%})")
    print(f"  Top-3 : {test_sum['top3']}/{test_sum['n']} = {test_sum['top3']/test_sum['n']:.1%}")
    print(f"  Domain: {test_sum['domain']}/{test_sum['n']} = {test_sum['domain']/test_sum['n']:.1%}")
    print(f"  과적합 폭: {train_sum['top1']/train_sum['n'] - test_sum['top1']/test_sum['n']:+.1%}")

    fields = list(test_rows[0].keys())
    with open(args.outdir / f"{tag}_test_predictions.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(test_rows)
    with open(args.outdir / f"{tag}_train_predictions.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(train_rows)

    curve = coverage_curve(test_rows)
    with open(args.outdir / f"{tag}_coverage_curve.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)

    (args.outdir / f"{tag}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.outdir}/{tag}_*.csv, {tag}_result.json")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataError as error:
        print(f"[오류] {error}")
        raise SystemExit(1)
