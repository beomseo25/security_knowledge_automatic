"""카테고리 profile CSV를 Neo4j Category 노드의 접두사 붙은 속성으로 적재한다.

예) --prefix nist_r1_ 이면 Category.nist_r1_detail_description / nist_r1_keywords /
nist_r1_exclusions 를 설정한다. 기존 속성(접두사 없는 deploy profile, 다른 접두사)은
건드리지 않는다. 노드를 만들거나 지우지 않고, 이미 있는 Category(id)에 속성만 더한다.

접속 정보는 algorithm/.env, NEO4J_* 환경변수가 있으면 그것이 우선한다.

사용법 (-final 저장소 루트에서):
    NEO4J_PASSWORD=... python algorithm/load_profile_to_neo4j.py \\
        --csv data/category_profiles_nist_tuned_r1.csv --prefix nist_r1_
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("detail_description", "keywords", "exclusions")


def main() -> int:
    ap = argparse.ArgumentParser(description="profile CSV를 Category 속성(접두사)으로 적재")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--prefix", required=True, help="예: nist_r1_ (영문·숫자·밑줄만)")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*_", args.prefix):
        print("[오류] --prefix 는 영문으로 시작하고 밑줄로 끝나야 합니다. 예: nist_r1_")
        return 1

    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / "algorithm" / ".env", override=False)

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 67:
        print("[오류] profile이 67개가 아니라 %d개입니다." % len(rows))
        return 1

    query = ("UNWIND $rows AS r MATCH (c:Category {id: r.category_id}) "
             "SET c.%sdetail_description = r.detail_description, "
             "c.%skeywords = r.keywords, c.%sexclusions = r.exclusions "
             "RETURN count(c) AS n" % ((args.prefix,) * 3))
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    try:
        driver.verify_connectivity()
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            n = session.run(query, rows=[{k: r[k] for k in ("category_id",) + FIELDS} for r in rows]).single()["n"]
    finally:
        driver.close()
    print("적재: %d개 Category에 %s* 속성을 설정했습니다." % (n, args.prefix))
    return 0 if n == 67 else 1


if __name__ == "__main__":
    sys.exit(main())
