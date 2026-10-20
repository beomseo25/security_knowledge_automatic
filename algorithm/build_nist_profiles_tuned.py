"""[3번 방식] 22건의 오답을 보고 NIST profile을 라운드별로 다듬는다 (in-sample 튜닝).

- 라운드 N 의 입력은 라운드 N-1 의 profile 파일이다 (라운드 0 = data/category_profiles_nist.csv).
- 출력은 항상 새 파일 data/category_profiles_nist_tuned_r{N}.csv 이고, 있으면 덮어쓰지 않는다.
  그래서 모든 시도가 파일로 남는다.
- 변경분(DELTAS)은 이 파일에 그대로 적혀 있다. 무엇을 왜 바꿨는지는 NIST_TUNING_LOG.md 에 기록한다.

정직성 표시: 이 방식은 채점에 쓰는 22건의 오답을 보고 설명서를 고친다. 따라서 같은 22건으로 잰
일치율은 일반화 성능이 아니라 '들어맞게 맞춘 값'이다. 보고할 때 반드시 그렇게 표기한다.

사용법 (-final 저장소 루트에서):
    python algorithm/build_nist_profiles_tuned.py --round 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE_R0 = DATA / "category_profiles_nist.csv"

# 라운드 -> 카테고리 -> {"add_desc": 설명에 덧붙일 영어 문장, "add_kw": 덧붙일 키워드,
#                        "drop_kw": 지울 키워드, "drop_desc": 설명에서 지울 문장(이전 라운드에서 덧붙인 것)}
DELTAS: dict[int, dict[str, dict]] = {
    1: {
        # GV.OV-01 오답: 임베딩은 GOV-03을 1위로 뽑았으나 LLM이 risk management strategy 키워드가 있는 RSK-01로 바꿨다.
        "GOV-03": {"add_desc": "Results of the risk management strategy are reviewed to inform and adjust the strategy and direction of the program.",
                   "add_kw": "review results of the strategy, adjust strategy and direction, strategy outcomes reviewed"},
        "RSK-01": {"drop_kw": ["risk management strategy"]},
        # GV.SC-01 오답: v1에서 공급망 프로그램·전략 어휘를 SUP-01에 붙였으나 의미상 SUP-02(공급망 관리) 쪽이다.
        "SUP-01": {"drop_kw": ["supply chain risk management", "supply chain risk strategy"]},
        "SUP-02": {"add_desc": "A supply chain risk management program, strategy, objectives, policies and processes are established and agreed to by organizational stakeholders.",
                   "add_kw": "supply chain risk management, supply chain risk management program, supply chain strategy, supply chain objectives, supply chain policies and processes"},
        # RC.RP-01 오답: 정답 BCP-01(임베딩 9위). 가장 억지스러운 수정 — 복구 실행 어휘를 BCP-01에 붙인다.
        "BCP-01": {"add_desc": "Recovery actions are executed once initiated to restore operations and data after an incident.",
                   "add_kw": "recovery execution, execute recovery, restore operations after incident, recovery actions"},
    },
    2: {
        # r1 결과: RC.CO-03이 BCP-01로 끌려가는 부작용(r1의 BCP-01 복구 어휘 때문). BCP-01을 r0 상태로 되돌린다.
        # RC.RP-01(정답 BCP-01)은 맞추려 하면 다른 항목이 깨지므로 맞추지 않는다.
        "BCP-01": {"drop_desc": "Recovery actions are executed once initiated to restore operations and data after an incident.",
                   "drop_kw": ["recovery execution", "execute recovery", "restore operations after incident", "recovery actions"]},
        # RC.CO-03 오답: 복구 활동 소통 어휘를 INC-02(보고·통지)에 붙인다.
        "INC-02": {"add_desc": "Recovery progress and restoration activities are communicated to internal and external stakeholders.",
                   "add_kw": "recovery communication, communicate restoration activities, public updates, recovery progress"},
        # RS.MA-01 오답: INC-01(계획 수립)로 갔다. 계획을 '실행'하는 쪽 어휘를 INC-03에 보강한다.
        "INC-03": {"add_desc": "The incident response plan is executed once an incident is declared.",
                   "add_kw": "incident response plan executed, once an incident is declared, execute plan"},
    },
}


def apply(rows: list[dict], delta: dict[str, dict]) -> list[dict]:
    by_id = {r["category_id"]: r for r in rows}
    unknown = [c for c in delta if c not in by_id]
    if unknown:
        raise SystemExit("[오류] 모르는 카테고리: %s" % unknown)
    for cid, change in delta.items():
        r = by_id[cid]
        if change.get("drop_desc"):
            if (" " + change["drop_desc"]) not in r["detail_description"]:
                raise SystemExit("[오류] %s 설명에 지울 문장이 없습니다." % cid)
            r["detail_description"] = r["detail_description"].replace(" " + change["drop_desc"], "", 1)
        if change.get("add_desc"):
            r["detail_description"] = r["detail_description"].rstrip() + " " + change["add_desc"]
        kws = [k.strip() for k in r["keywords"].split(",") if k.strip()]
        for k in change.get("drop_kw", []):
            if k not in kws:
                raise SystemExit("[오류] %s 키워드에 '%s' 가 없습니다." % (cid, k))
            kws.remove(k)
        kws += [k.strip() for k in change.get("add_kw", "").split(",") if k.strip()]
        r["keywords"] = ", ".join(kws)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="NIST profile 라운드별 튜닝 (3번 방식)")
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    if args.round not in DELTAS:
        print("[오류] 라운드 %d 의 변경분이 DELTAS 에 없습니다." % args.round)
        return 1
    src = BASE_R0 if args.round == 1 else DATA / ("category_profiles_nist_tuned_r%d.csv" % (args.round - 1))
    dst = DATA / ("category_profiles_nist_tuned_r%d.csv" % args.round)
    if dst.exists():
        print("[안내] %s 가 이미 있습니다. 덮어쓰지 않습니다." % dst)
        return 1
    with open(src, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    rows = apply(rows, DELTAS[args.round])
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("저장: %s (입력 %s, 변경 카테고리 %s)" % (dst.name, src.name, sorted(DELTAS[args.round])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
