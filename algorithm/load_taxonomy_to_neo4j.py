"""data/category_profiles_deploy.csv 로 Neo4j 에 Domain 16개 / Category 67개 / HAS_CATEGORY 67개를 적재한다.

- MERGE 만 쓴다. 노드·관계를 지우지 않으므로 다시 실행해도 같은 상태가 된다(멱등).
- Category 에는 접두사 없는 deploy profile 속성(detail_description, keywords, exclusions)이 들어간다.
  NIST 용 profile 은 load_profile_to_neo4j.py 가 접두사 붙은 별도 속성(nist_, nist_r1_, nist_r2_)으로 더한다.
- 접속 정보는 algorithm/.env, NEO4J_* 환경변수가 있으면 그것이 우선한다.

사용법 (저장소 루트에서):
    NEO4J_PASSWORD=... python algorithm/load_taxonomy_to_neo4j.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "data" / "category_profiles_deploy.csv"

QUERY = """
UNWIND $rows AS r
MERGE (d:Domain {id: r.domain_id})
  SET d.name = r.domain_name, d.description = r.domain_description
MERGE (c:Category {id: r.category_id})
  SET c.name = r.category_name, c.detail_description = r.detail_description,
      c.keywords = r.keywords, c.exclusions = r.exclusions
MERGE (d)-[:HAS_CATEGORY]->(c)
"""


def main() -> int:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / "algorithm" / ".env", override=False)

    with open(PROFILES, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 67:
        print("[오류] profile이 67개가 아니라 %d개입니다." % len(rows))
        return 1

    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    try:
        driver.verify_connectivity()
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            session.run("CREATE CONSTRAINT domain_id_unique IF NOT EXISTS FOR (n:Domain) REQUIRE n.id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT category_id_unique IF NOT EXISTS FOR (n:Category) REQUIRE n.id IS UNIQUE").consume()
            session.run(QUERY, rows=rows).consume()
            counts = session.run("MATCH (d:Domain) WITH count(d) AS domains "
                                 "MATCH (c:Category) WITH domains, count(c) AS categories "
                                 "MATCH ()-[r:HAS_CATEGORY]->() RETURN domains, categories, count(r) AS rels").single()
    finally:
        driver.close()
    print("적재 완료: Domain %d, Category %d, HAS_CATEGORY %d" % (counts["domains"], counts["categories"], counts["rels"]))
    return 0 if (counts["domains"], counts["categories"], counts["rels"]) == (16, 67, 67) else 1


if __name__ == "__main__":
    sys.exit(main())
