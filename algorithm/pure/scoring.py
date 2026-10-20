"""배포용 점수 계산 — 하네스와 **같은 공식**을 쓰는 얇은 부품.

`benchmark_pure.py` 는 config 469개를 전부 만들지만, 배포 프로그램은 그중 하나만
쓰면 된다. 이 모듈은 그 하나를 계산한다. 공식은 하네스에서 그대로 옮겨 왔고,
`tests_scoring_matches_harness.py` 가 두 경로의 결과가 소수점까지 같은지 확인한다.

지원하는 config 이름 형태
  - `<카테고리문장>__grid__d<가중치>_s<가중치>_c<가중치>`  (행 단위 min-max 후 가중합)
  - `<카테고리문장>__official_0.4_0.2_0.4`                (정규화 없이 고정 가중합)
  - `<카테고리문장>__dense` / `__sparse` / `__colbert`     (한 가지만)

카테고리 문장 종류는 `benchmark_lofo.build_category_text_groups` 와 같다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ALGO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ALGO))

from benchmark_lofo import (  # noqa: E402
    build_category_text_groups,
    colbert_matrix,
    copy_model_output,
    minmax_rows,
)

GRID = re.compile(r"^(?P<cat>[a-z_]+)__grid__d(?P<d>[\d.]+)_s(?P<s>[\d.]+)_c(?P<c>[\d.]+)$")
OFFICIAL = re.compile(r"^(?P<cat>[a-z_]+)__official_0\.4_0\.2_0\.4$")
SINGLE = re.compile(r"^(?P<cat>[a-z_]+)__(?P<mode>dense|sparse|colbert)$")


class ConfigError(Exception):
    pass


def parse_config(name: str) -> tuple[str, str, tuple[float, float, float]]:
    """config 이름을 (카테고리 문장 종류, 방식, 가중치) 로 푼다."""
    m = GRID.match(name)
    if m:
        return m.group("cat"), "grid", (float(m.group("d")), float(m.group("s")),
                                        float(m.group("c")))
    m = OFFICIAL.match(name)
    if m:
        return m.group("cat"), "official", (0.4, 0.2, 0.4)
    m = SINGLE.match(name)
    if m:
        weights = {"dense": (1.0, 0.0, 0.0), "sparse": (0.0, 1.0, 0.0),
                   "colbert": (0.0, 0.0, 1.0)}[m.group("mode")]
        return m.group("cat"), "single", weights
    raise ConfigError(
        "배포 프로그램이 계산할 수 있는 config 이름이 아닙니다: %s\n"
        "  (지원: __grid__d._s._c. / __official_0.4_0.2_0.4 / __dense|__sparse|__colbert)" % name)


def encode(model, texts: list[str], batch_size: int = 8, max_length: int = 256) -> dict:
    """하네스와 같은 인코딩 설정."""
    return copy_model_output(model.encode(
        texts, batch_size=batch_size, max_length=max_length,
        return_dense=True, return_sparse=True, return_colbert_vecs=True))


def category_texts_for(profiles: list[dict], config_name: str) -> list[str]:
    group, _, _ = parse_config(config_name)
    groups = build_category_text_groups(profiles)
    if group not in groups:
        raise ConfigError("모르는 카테고리 문장 종류: %s (가능: %s)"
                          % (group, ", ".join(groups)))
    return groups[group]


def score(model, item_out: dict, cat_out: dict, config_name: str) -> np.ndarray:
    """(항목 수 x 카테고리 수) 점수 행렬. 하네스와 같은 공식."""
    _, kind, weights = parse_config(config_name)
    dense = np.einsum("id,jd->ij", item_out["dense_vecs"], cat_out["dense_vecs"],
                      optimize=False).astype(np.float32)
    sparse = model.compute_lexical_matching_score(
        item_out["lexical_weights"], cat_out["lexical_weights"]).astype(np.float32)
    colbert = colbert_matrix(item_out["colbert_vecs"], cat_out["colbert_vecs"])
    if kind == "grid":
        parts = (minmax_rows(dense), minmax_rows(sparse), minmax_rows(colbert))
    else:
        parts = (dense, sparse, colbert)
    return sum(w * m for w, m in zip(weights, parts))
