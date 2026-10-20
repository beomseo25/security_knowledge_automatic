"""NIST CSF 2.0 22개에 85.5% 파이프라인(embedding + 마진 하위 구간 LLM 재판정)을 그대로 적용한다.

재사용 (수정하지 않음):
- `classify.classify()` — 배포본 본체. 점수 계산, 마진 경로 판정, LLM 프롬프트(`LLM_PROMPT`),
  후보 섞기(seed=항목 순번), LLM 답 반영 규칙이 전부 여기 있다.
- `algorithm/pure/classify_config.json` — config `profile_full__grid__d0.1_s0.2_c0.7`,
  설명서 `data/category_profiles_deploy.csv`, 임계값 margin_llm=0.202556, K=15.

함께 재는 대조군 (가설 A 단독, LLM 없음):
- rich+dense  : 같은 설명서, `profile_full__dense`
- rich+hybrid : 같은 설명서, 위 config의 embedding 1위 (= classify의 cand1)

규칙:
- 입력은 문장 하나뿐(text_en). NIST ID·정답은 프롬프트에 들어가지 않는다.
- `--dry-run` 은 가짜 LLM으로 코드 경로와 프롬프트 유출 검사만 하고, 결과는 별도 폴더에 쓴다.
- 기존 결과 파일은 덮어쓰지 않는다.

사용법 (-final 저장소 루트에서):
    python algorithm/nist_full_pipeline.py --input <nist csv> --dry-run
    python algorithm/nist_full_pipeline.py --input <nist csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import classify as deploy  # noqa: E402  (classify.py가 algorithm/, pure/, llm/ 경로를 잡는다)
from benchmark_lofo import load_profiles  # noqa: E402
from llm_client import FakeLLM, LLMClient  # noqa: E402
from scoring import category_texts_for, encode, score  # noqa: E402

RESULTS = ROOT / "algorithm" / "results"
FOCUS_IDS = ("RS.MA-01", "RS.AN-03", "RS.MI-01")


def read_rows(path: Path, text_column: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for need in (text_column, "answer_category_id", "nist_id"):
        if need not in rows[0]:
            raise SystemExit("[오류] 입력 CSV에 '%s' 열이 없습니다." % need)
    return rows


def audit_prompts(prompts: list[tuple[str, str]], rows: list[dict], sentences: list[str],
                  ranked_by_sentence: dict[str, list[str]]) -> list[str]:
    """rejudge_pure.audit_prompt 와 같은 항목을 NIST 입력에 맞춰 검사한다."""
    problems = []
    nist_ids = [r["nist_id"] for r in rows]
    for _, prompt in prompts:
        item = prompt.split("[후보 분류]", 1)[0]
        sentence = next((s for s in sentences if ("제목: " + s.strip() + "\n") in item), None)
        if sentence is None:
            problems.append("항목 블록이 '제목: <문장>' 이 아니다")
            continue
        for nid in nist_ids:
            if nid in prompt:
                problems.append("NIST ID %s" % nid)
        if re.search(r"\d+\.\d{3,}", prompt):
            problems.append("점수로 보이는 소수 숫자")
        if "마진" in prompt or "점수" in item or "순위" in item or "1위" in prompt:
            problems.append("'마진'/'점수'/'순위' 표현")
        for marker in ("정답", "answer", "gold", "correct"):
            if marker in prompt:
                problems.append("정답 표시 '%s'" % marker)
        shown = re.findall(r"category_id: (\S+)", prompt)
        if shown == ranked_by_sentence[sentence][:len(shown)]:
            problems.append("후보 순서가 embedding 순위 그대로다")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="NIST 22개 — 85.5% 파이프라인 그대로 적용")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--text-column", default="text_en")
    ap.add_argument("--dry-run", action="store_true", help="가짜 LLM으로 코드·유출 검사만")
    ap.add_argument("--config", type=Path, default=deploy.CONFIG_PATH)
    args = ap.parse_args()

    out_dir = RESULTS / ("full_pipeline_dryrun" if args.dry_run else "full_pipeline")
    out_csv = out_dir / "nist_full_pipeline.csv"
    if out_csv.exists():
        print("[안내] %s 가 이미 있습니다. 덮어쓰지 않습니다." % out_csv)
        return 1

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_rows(args.input, args.text_column)
    sentences = [r[args.text_column].strip() for r in rows]
    k = int(settings["llm_candidates"])
    t_llm = float(settings["margin_llm"])

    # --- 모델을 한 번만 올려 대조군 계산과 classify() 양쪽에서 같이 쓴다 ---
    from FlagEmbedding import BGEM3FlagModel
    import FlagEmbedding
    model = BGEM3FlagModel(deploy.MODEL_NAME, use_fp16=False)  # classify.py 와 같은 fp32
    FlagEmbedding.BGEM3FlagModel = lambda *a, **kw: model  # classify() 안의 재로딩을 막는다

    profiles = load_profiles(ROOT / settings["profiles_csv"])
    cat_ids = [p["category_id"] for p in profiles]
    item_out = encode(model, sentences)

    def ranked_ids(config_name: str) -> np.ndarray:
        cat_out = encode(model, category_texts_for(profiles, config_name))
        return np.argsort(-score(model, item_out, cat_out, config_name), axis=1, kind="stable")

    dense_order = ranked_ids("profile_full__dense")
    hybrid_order = ranked_ids(settings["config"])
    ranked_by_sentence = {s: [cat_ids[j] for j in hybrid_order[i]] for i, s in enumerate(sentences)}

    # --- LLM 클라이언트: 가짜 또는 실제. 보낸 프롬프트를 기록해 둔다 ---
    if args.dry_run:
        def fake(prompt, system):
            first = re.search(r"category_id: (\S+)", prompt).group(1)
            return '{"category_id": "%s", "reason": "가짜 응답"}' % first
        client = FakeLLM(fake)
    else:
        client = LLMClient()
        sent: list[tuple[str, str]] = []
        real_ask = client.ask

        def recording_ask(prompt, system=""):
            sent.append((system, prompt))
            return real_ask(prompt, system)
        client.ask = recording_ask
    deploy.make_llm = lambda quiet=False: (client, "")
    print("LLM: %s / %s" % (client.provider, client.model))

    results = deploy.classify(sentences, settings, use_llm=True, quiet=True)

    prompts = client.prompts if args.dry_run else sent
    problems = audit_prompts(prompts, rows, sentences, ranked_by_sentence)
    print("프롬프트 유출 검사: %d개 중 위반 %d건 %s"
          % (len(prompts), len(problems), problems[:5] if problems else "— 깨끗함"))
    if problems:
        print("[중단] 유출 위반이 있어 결과를 저장하지 않습니다.")
        return 1

    out_rows = []
    for i, (row, res) in enumerate(zip(rows, results)):
        gold = row["answer_category_id"].strip()
        emb_top = [cat_ids[j] for j in hybrid_order[i]]
        dense_top = [cat_ids[j] for j in dense_order[i]]
        assert res["cand1"].split()[0] == emb_top[0], "classify()와 대조군 순위가 다르다"
        ambiguous = res["margin"] < t_llm
        out_rows.append({
            "no": row.get("no", i + 1), "nist_id": row["nist_id"], "input_sentence": sentences[i],
            "answer_category_id": gold,
            "rich_dense_top1": dense_top[0], "rich_dense_top3": "|".join(dense_top[:3]),
            "rich_dense_top1_match": dense_top[0] == gold, "rich_dense_top3_match": gold in dense_top[:3],
            "emb_top1": emb_top[0], "emb_top2": emb_top[1], "emb_top3": emb_top[2],
            "emb_top1_match": emb_top[0] == gold, "emb_top3_match": gold in emb_top[:3],
            "gold_rank": emb_top.index(gold) + 1 if gold in emb_top else "",
            "gold_in_llm_candidates": gold in emb_top[:k],
            "margin": res["margin"], "ambiguous": ambiguous, "route": res["route"],
            "llm_used": res["llm_used"], "final_category_id": res["category_id"],
            "final_match": res["category_id"] == gold,
            "changed_by_llm": res["llm_used"] and res["category_id"] != emb_top[0],
            "llm_reason": res["llm_reason"], "note": res["note"],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    def pct(key): return "%d/%d (%.1f%%)" % (sum(r[key] for r in out_rows), n,
                                             100 * sum(r[key] for r in out_rows) / n)
    print("\n=== 결과 (n=%d, 임계값 margin<%.6f, K=%d) ===" % (n, t_llm, k))
    print("rich+dense        Top-1 %s  Top-3 %s" % (pct("rich_dense_top1_match"), pct("rich_dense_top3_match")))
    print("rich+hybrid       Top-1 %s  Top-3 %s" % (pct("emb_top1_match"), pct("emb_top3_match")))
    print("rich+hybrid+LLM   최종  %s" % pct("final_match"))
    llm_rows = [r for r in out_rows if r["llm_used"]]
    fixed = sum(1 for r in llm_rows if r["final_match"] and not r["emb_top1_match"])
    broken = sum(1 for r in llm_rows if not r["final_match"] and r["emb_top1_match"])
    print("LLM 재판정 %d개: 오답→정답 %d, 정답→오답 %d, 정답이 후보 15개 안 %d개"
          % (len(llm_rows), fixed, broken, sum(r["gold_in_llm_candidates"] for r in llm_rows)))
    print("\n=== 항목별 ===")
    for r in out_rows:
        mark = "*" if r["nist_id"] in FOCUS_IDS else " "
        print("%s%-9s 정답 %-7s emb %-7s(정답%2s위) margin %.4f %-14s 최종 %-7s %s"
              % (mark, r["nist_id"], r["answer_category_id"], r["emb_top1"], r["gold_rank"],
                 r["margin"], r["route"], r["final_category_id"], "O" if r["final_match"] else "X"))
    print("\n저장: %s" % out_csv)
    print("LLM 사용량: %s" % client.usage.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
