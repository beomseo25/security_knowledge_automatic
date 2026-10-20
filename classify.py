#!/usr/bin/env python
# 원본 출처: security-knowledge-graph-neo4j-final/classify.py (배포용 분류기, 내용 수정 없이 그대로 복사).
# 이 파일 상단의 이 주석 세 줄만 추가했다. 원본 저장소의 경로 구조(algorithm/pure, algorithm/llm)에 의존한다.
# NIST 22건 실험에서는 nist_llm_hybrid_prototype.py 가 classify() 를 그대로 불러 쓴다.
"""통제항목 문장 하나를 보안 분류체계 67개 Category 중 하나에 배정한다.

순수 입력 규칙
--------------
**입력은 통제항목 문장 하나뿐이다.** 프레임워크 이름도, 상위분류도, 상세문장도,
통제항목 번호도 요구하지 않는다. CSV로 넣을 때도 문장이 든 칸 하나만 있으면 된다.

쓰는 법
-------
    # 문장 하나
    python classify.py "퇴사자 계정을 퇴사 당일에 비활성화하고 그 결과를 기록한다"

    # 여러 문장 (CSV — 문장 칸 하나만 있으면 된다)
    python classify.py --input items.csv --output result.csv

    # LLM 없이 embedding 단독으로
    python classify.py --input items.csv --no-llm

출력
----
항목마다 배정 Category(id·이름·상위 도메인), 마진과 처리 경로, 상위 3개 후보와
점수, LLM이 판정했으면 그 이유 한 줄을 낸다. `--output` 을 주면 CSV로도 남긴다.

처리 경로
---------
    [자동확정]      마진이 충분히 크다. 그대로 쓴다.
    [LLM재판정]     마진이 작다. 상위 후보 몇 개를 LLM에게 넘겨 다시 고르게 한다.
    [사람검토 권고] 마진이 아주 작거나 LLM도 확신하지 못한다. 사람이 봐야 한다.

임계값과 config는 `algorithm/pure/classify_config.json` 에 있고, 그 값이 어디서
나왔는지도 그 파일에 적혀 있다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ALGO = ROOT / "algorithm"
sys.path.insert(0, str(ALGO))
sys.path.insert(0, str(ALGO / "pure"))
sys.path.insert(0, str(ALGO / "llm"))

from benchmark_lofo import load_profiles  # noqa: E402
from scoring import category_texts_for, encode, score  # noqa: E402

CONFIG_PATH = ALGO / "pure" / "classify_config.json"
CACHE_DIR = ROOT / "data" / "deploy_cache"
MODEL_NAME = "BAAI/bge-m3"

# CSV에서 문장 칸으로 인정하는 이름들 (칸이 하나뿐이면 이름은 아무래도 좋다)
SENTENCE_COLUMNS = ("title", "sentence", "text", "item", "문장", "통제항목", "항목", "제목")

ROUTE_AUTO = "[자동확정]"
ROUTE_LLM = "[LLM재판정]"
ROUTE_HUMAN = "[사람검토 권고]"

OUT_FIELDS = ["no", "input_sentence", "category_id", "category_name", "domain_id",
              "domain_name", "margin", "route", "decided_by", "top1_score",
              "cand1", "cand1_score", "cand2", "cand2_score", "cand3", "cand3_score",
              "llm_used", "llm_reason", "note"]

LLM_SYSTEM = (
    "당신은 정보보호 통제 항목을 사내 보안 분류 체계에 배정하는 보안 컨설턴트입니다. "
    "주어진 후보 분류 중에서 항목의 의미에 가장 잘 맞는 것 하나만 고릅니다. "
    "반드시 JSON 한 줄로만 답합니다."
)

LLM_PROMPT = """다음은 어떤 조직의 정보보호 통제 항목 하나입니다.

[통제 항목]
제목: {sentence}

이 항목을 아래 후보 분류 {n} 개 중 **딱 하나**에 배정하세요.

[후보 분류]
{candidates}

판단 기준:
- 항목이 실제로 요구하는 활동이 무엇인지 보고, 그 활동을 다루는 분류를 고르세요.
- 후보 목록의 순서에는 아무 의미가 없습니다.
- 반드시 위 후보의 category_id 중 하나만 고르세요. 목록에 없는 값은 안 됩니다.

아래 형식의 JSON 한 줄로만 답하세요. 다른 말은 쓰지 마세요.
{{"category_id": "<고른 id>", "reason": "<한 문장 근거>"}}"""


class InputError(Exception):
    pass


# ---------------------------------------------------------------------------
# 입력 읽기 — 문장 외에 아무 칸도 요구하지 않는다
# ---------------------------------------------------------------------------

def read_sentences(path: Path) -> list[str]:
    if path.suffix.lower() in (".txt", ".text"):
        lines = [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines()]
        return [l for l in lines if l]

    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise InputError("%s: 빈 파일입니다." % path)

    header = [h.strip() for h in rows[0]]
    lowered = [h.lower() for h in header]
    column = None
    if len(header) == 1:
        # 칸이 하나뿐이면 이름은 상관없다. 머리글인지 데이터인지만 가린다.
        column = 0
        body = rows[1:] if lowered[0] in [c.lower() for c in SENTENCE_COLUMNS] else rows
        return [r[0].strip() for r in body if r and r[0].strip()]
    for i, name in enumerate(lowered):
        if name in [c.lower() for c in SENTENCE_COLUMNS]:
            column = i
            break
    if column is None:
        raise InputError(
            "%s: 어느 칸이 문장인지 알 수 없습니다.\n"
            "  칸 이름: %s\n"
            "  칸을 하나만 두거나, 문장 칸 이름을 다음 중 하나로 해 주세요: %s"
            % (path, ", ".join(header), ", ".join(SENTENCE_COLUMNS)))
    return [r[column].strip() for r in rows[1:] if len(r) > column and r[column].strip()]


# ---------------------------------------------------------------------------
# 카테고리 인코딩 캐시 — 매번 다시 인코딩하지 않는다
# ---------------------------------------------------------------------------

def cache_key(profiles_path: Path, config_name: str) -> str:
    import hashlib
    digest = hashlib.sha256(profiles_path.read_bytes()).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", config_name)
    return "%s__%s" % (digest, safe)


def load_or_encode_categories(model_getter, profiles: list[dict], profiles_path: Path,
                              config_name: str, quiet: bool = False) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / ("categories_%s.npz" % cache_key(profiles_path, config_name))
    if path.exists():
        blob = np.load(path, allow_pickle=True)
        return {"dense_vecs": blob["dense"],
                "lexical_weights": list(blob["lexical"]),
                "colbert_vecs": list(blob["colbert"])}
    model = model_getter()  # 모델을 먼저 불러온 뒤에 안내를 찍는다 (메시지 순서)
    if not quiet:
        print("카테고리 %d개를 인코딩합니다 (처음 한 번만, 이후에는 캐시를 씁니다)…"
              % len(profiles), flush=True)
    out = encode(model, category_texts_for(profiles, config_name))
    np.savez(path,
             dense=out["dense_vecs"],
             lexical=np.array(out["lexical_weights"], dtype=object),
             colbert=np.array(out["colbert_vecs"], dtype=object))
    return out


# ---------------------------------------------------------------------------
# LLM 재판정 (없어도 동작한다)
# ---------------------------------------------------------------------------

def make_llm(quiet: bool = False):
    """LLM 클라이언트를 만든다. 키가 없거나 실패하면 (None, 이유) 를 준다."""
    try:
        from llm_client import LLMClient
        return LLMClient(), ""
    except Exception as error:
        reason = str(error).splitlines()[0]
        if not quiet:
            print("[안내] LLM을 쓸 수 없습니다 — embedding 단독으로 진행합니다. (%s)" % reason)
        return None, reason


def llm_pick(client, sentence: str, candidates: list[dict], seed: int) -> tuple[str, str, str]:
    """(고른 id, 이유, 오류) — 실패하면 고른 id가 빈 문자열이다."""
    import random

    from llm_client import extract_json
    shown = list(candidates)
    random.Random(seed).shuffle(shown)
    blocks = "\n".join(
        "- category_id: %s\n  이름: %s > %s\n  설명: %s\n  대표 키워드: %s"
        % (c["category_id"], c["domain_name"], c["category_name"],
           c["detail_description"], c["keywords"]) for c in shown)
    prompt = LLM_PROMPT.format(sentence=sentence.strip(), n=len(shown), candidates=blocks)
    try:
        raw = client.ask(prompt, LLM_SYSTEM)
        parsed = extract_json(raw)
        pick = str(parsed.get("category_id", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
    except Exception as error:
        return "", "", str(error).splitlines()[0]
    valid = {c["category_id"] for c in shown}
    if pick not in valid:
        return "", reason, "LLM이 후보에 없는 id를 골랐습니다: %r" % pick
    return pick, reason, ""


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def classify(sentences: list[str], settings: dict, use_llm: bool = True,
             quiet: bool = False) -> list[dict]:
    profiles_path = ROOT / settings["profiles_csv"]
    config_name = settings["config"]
    profiles = load_profiles(profiles_path)
    by_id = {p["category_id"]: p for p in profiles}
    cat_ids = [p["category_id"] for p in profiles]

    model_box: list = []

    def get_model():
        if not model_box:
            from FlagEmbedding import BGEM3FlagModel
            if not quiet:
                print("모델을 불러옵니다: %s (처음 한 번은 시간이 걸립니다)…"
                      % MODEL_NAME, flush=True)
            model_box.append(BGEM3FlagModel(MODEL_NAME, use_fp16=False))
        return model_box[0]

    cat_out = load_or_encode_categories(get_model, profiles, profiles_path, config_name, quiet)
    item_out = encode(get_model(), sentences)
    scores = score(get_model(), item_out, cat_out, config_name)

    client, llm_off_reason = (None, "--no-llm 으로 껐습니다")
    if use_llm:
        client, llm_off_reason = make_llm(quiet)

    t_llm = float(settings["margin_llm"])
    t_human = float(settings["margin_human"])
    k = int(settings["llm_candidates"])

    order = np.argsort(-scores, axis=1, kind="stable")
    results = []
    for i, sentence in enumerate(sentences):
        ranked = order[i]
        top = [cat_ids[j] for j in ranked[:max(3, k)]]
        s1, s2 = float(scores[i, ranked[0]]), float(scores[i, ranked[1]])
        margin = s1 - s2
        chosen, decided_by, reason, note, llm_used = top[0], "embedding", "", "", False

        if margin >= t_llm:
            route = ROUTE_AUTO
        else:
            route = ROUTE_LLM
            if client is None:
                route = ROUTE_HUMAN
                note = "LLM 미사용 (%s) — embedding 1위를 잠정 답으로 둔다" % llm_off_reason
            else:
                cands = [by_id[c] for c in top[:k]]
                pick, reason, error = llm_pick(client, sentence, cands, seed=i)
                llm_used = True
                if pick:
                    chosen, decided_by = pick, "llm"
                else:
                    route = ROUTE_HUMAN
                    note = "LLM 호출 실패 — embedding 1위를 잠정 답으로 둔다 (%s)" % error
            if margin < t_human:
                route = ROUTE_HUMAN
                if not note:
                    # LLM 답은 남겨 두되 사람이 확인하도록 표시한다. 이 구간에서
                    # LLM 정확도는 약 75%로, 자동확정 구간(약 87%)보다 눈에 띄게 낮다.
                    note = "마진이 아주 작다 — %s의 잠정 답이니 사람이 확인할 것" % decided_by

        p = by_id[chosen]
        row = {"no": i + 1, "input_sentence": sentence,
               "category_id": chosen, "category_name": p["category_name"],
               "domain_id": p["domain_id"], "domain_name": p["domain_name"],
               "margin": round(margin, 6), "route": route,
               # decided_by 는 "이 배정을 사람이 봐야 하는가"를 추적하기 위한 칸이다.
               # 사람 검토 구간이면 human_review 로 두고, 잠정 답을 낸 주체는
               # llm_used·note 에 남긴다.
               "decided_by": "human_review" if route == ROUTE_HUMAN else decided_by,
               "top1_score": round(s1, 6),
               "llm_used": llm_used, "llm_reason": reason, "note": note}
        for n, j in enumerate(ranked[:3], start=1):
            cid = cat_ids[j]
            row["cand%d" % n] = "%s %s" % (cid, by_id[cid]["category_name"])
            row["cand%d_score" % n] = round(float(scores[i, j]), 6)
        results.append(row)
    return results


def print_result(row: dict) -> None:
    print("=" * 74)
    print("[%d] %s" % (row["no"], row["input_sentence"]))
    print("-" * 74)
    print("  배정      : %s  %s   (상위 도메인: %s %s)"
          % (row["category_id"], row["category_name"], row["domain_id"], row["domain_name"]))
    print("  신뢰도    : 마진 %.4f   ->  %s" % (row["margin"], row["route"]))
    print("  판정 주체 : %s%s" % (row["decided_by"], "" if row["llm_used"] else " (LLM 미사용)"))
    print("  근거      : 상위 3개 후보")
    for n in (1, 2, 3):
        print("      %d위  %-46s %.4f" % (n, row["cand%d" % n], row["cand%d_score" % n]))
    if row["llm_reason"]:
        print("  LLM 이유  : %s" % row["llm_reason"])
    if row["note"]:
        print("  비고      : %s" % row["note"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="통제항목 문장 하나를 보안 분류체계 Category에 배정한다 (입력은 문장뿐)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예)  python classify.py \"퇴사자 계정을 퇴사 당일에 비활성화한다\"\n"
               "     python classify.py --input items.csv --output result.csv")
    ap.add_argument("sentence", nargs="*", help="통제항목 문장 (여러 개 가능)")
    ap.add_argument("--input", type=Path, help="문장 목록 CSV (문장 칸 하나만 있으면 된다)")
    ap.add_argument("--output", type=Path, help="결과를 CSV로 저장할 경로")
    ap.add_argument("--no-llm", action="store_true", help="LLM을 쓰지 않는다")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.config.exists():
        raise InputError("설정 파일이 없습니다: %s" % args.config)
    settings = json.loads(args.config.read_text(encoding="utf-8"))

    sentences = list(args.sentence)
    if args.input:
        sentences += read_sentences(args.input)
    if not sentences:
        ap.print_help()
        return 1

    started = time.time()
    results = classify(sentences, settings, use_llm=not args.no_llm, quiet=args.quiet)

    for row in results:
        print_result(row)
    print("=" * 74)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["route"]] = counts.get(r["route"], 0) + 1
    print("%d개 처리, %.1f초.  %s"
          % (len(results), time.time() - started,
             "  ".join("%s %d개" % (k, v) for k, v in sorted(counts.items()))))
    print("설정: config=%s  임계값 LLM<%.4f, 사람검토<%.4f  (근거: %s)"
          % (settings["config"], settings["margin_llm"], settings["margin_human"],
             args.config.name))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
            w.writeheader()
            w.writerows(results)
        print("저장: %s" % args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as error:
        print("[오류] %s" % error)
        raise SystemExit(1)
