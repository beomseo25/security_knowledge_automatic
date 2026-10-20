"""NIST CSF 2.0 22개 — 카테고리 profile을 Neo4j에서 직접 읽어 85.5% 파이프라인을 적용한다.

nist_full_pipeline.py 와의 차이는 하나뿐이다: 카테고리 설명서를 category_profiles_deploy.csv
대신 Neo4j(Domain-[:HAS_CATEGORY]->Category)에서 읽는다. 하이브리드 임베딩 + 마진 하위 구간
LLM 재판정은 classify.classify() 를 수정 없이 그대로 쓴다.

- Neo4j에서 읽은 profile은 results/neo4j_pipeline*/profiles_from_neo4j.csv 로 덤프해
  classify()가 그 경로를 읽게 한다. 카테고리 인코딩 캐시도 별도 폴더를 쓴다(기존 캐시 오염 방지).
- 읽기 전용: Neo4j에 쓰지 않는다. 기존 결과 파일은 덮어쓰지 않는다.
- 접속 정보는 algorithm/.env, NEO4J_* 환경변수가 있으면 그것이 우선한다.

사용법 (-final 저장소 루트에서):
    python algorithm/nist_llm_hybrid_prototype.py --input data/nist_ground_truth_22.csv --dry-run
    python algorithm/nist_llm_hybrid_prototype.py --input data/nist_ground_truth_22.csv
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


PROFILE_QUERY = """
MATCH (d:Domain)-[:HAS_CATEGORY]->(c:Category)
RETURN c.id AS category_id, c.name AS category_name,
       d.id AS domain_id, d.name AS domain_name, d.description AS domain_description,
       c.{p}detail_description AS detail_description, c.{p}keywords AS keywords,
       c.{p}exclusions AS exclusions
ORDER BY c.id
"""
PROFILE_FIELDS = ["category_id", "category_name", "domain_id", "domain_name",
                  "domain_description", "detail_description", "keywords", "exclusions"]


def fetch_profiles_from_neo4j(prefix: str = "") -> list[dict]:
    """Neo4j에서 카테고리 profile을 읽는다 (읽기 전용)."""
    import os
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / "algorithm" / ".env", override=False)  # 환경변수가 .env보다 우선
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    try:
        driver.verify_connectivity()
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            return session.execute_read(lambda tx: [dict(r) for r in tx.run(PROFILE_QUERY.format(p=prefix))])
    finally:
        driver.close()


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
    ap = argparse.ArgumentParser(description="NIST 22개 — profile=Neo4j")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--text-column", default="text_en")
    ap.add_argument("--dry-run", action="store_true", help="가짜 LLM으로 코드·유출 검사만")
    ap.add_argument("--profile-prefix", default="",
                    help="Neo4j Category 속성 접두사. 예: nist_ 이면 nist_detail_description 등을 읽는다")
    ap.add_argument("--config", type=Path, default=deploy.CONFIG_PATH)
    args = ap.parse_args()

    tag = "neo4j_pipeline" + ("_" + args.profile_prefix.strip("_") if args.profile_prefix else "")
    out_dir = RESULTS / (tag + "_dryrun" if args.dry_run else tag)
    out_csv = out_dir / "nist_neo4j_pipeline.csv"
    if out_csv.exists():
        print("[안내] %s 가 이미 있습니다. 덮어쓰지 않습니다." % out_csv)
        return 1

    settings = json.loads(args.config.read_text(encoding="utf-8"))

    # --- 카테고리 profile: Neo4j 에서 직접 ---
    neo_profiles = fetch_profiles_from_neo4j(args.profile_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_dump = out_dir / "profiles_from_neo4j.csv"
    with open(profiles_dump, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        w.writeheader()
        w.writerows({k: p[k] for k in PROFILE_FIELDS} for p in neo_profiles)
    m = re.fullmatch(r"nist_(r\d+_)?", args.profile_prefix)
    if m:   # nist_ -> category_profiles_nist.csv, nist_rN_ -> category_profiles_nist_tuned_rN.csv
        ref_csv = ROOT / ("data/category_profiles_nist_tuned_%s.csv" % m.group(1).rstrip("_") if m.group(1)
                          else "data/category_profiles_nist.csv")
    else:
        ref_csv = ROOT / settings["profiles_csv"]
    with open(ref_csv, encoding="utf-8-sig") as f:
        csv_profiles = {r["category_id"]: r for r in csv.DictReader(f)}
    diff = [p["category_id"] for p in neo_profiles
            if any((p[k] or "") != csv_profiles.get(p["category_id"], {}).get(k) for k in PROFILE_FIELDS)]
    print("Neo4j profile %d개 (deploy CSV와 내용이 다른 카테고리: %d개 %s)"
          % (len(neo_profiles), len(diff), diff[:5]))
    settings["profiles_csv"] = str(profiles_dump)   # ROOT / 절대경로 = 절대경로
    deploy.CACHE_DIR = out_dir / "deploy_cache"      # 기존 인코딩 캐시와 섞지 않는다
    rows = read_rows(args.input, args.text_column)
    sentences = [r[args.text_column].strip() for r in rows]
    k = int(settings["llm_candidates"])
    t_llm = float(settings["margin_llm"])

    # --- 모델을 한 번만 올려 대조군 계산과 classify() 양쪽에서 같이 쓴다 ---
    from FlagEmbedding import BGEM3FlagModel
    import FlagEmbedding
    model = BGEM3FlagModel(deploy.MODEL_NAME, use_fp16=False)  # classify.py 와 같은 fp32
    FlagEmbedding.BGEM3FlagModel = lambda *a, **kw: model  # classify() 안의 재로딩을 막는다

    profiles = load_profiles(Path(settings["profiles_csv"]))
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
