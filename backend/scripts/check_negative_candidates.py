"""校验候选负样本：确认库里真的没有能回答它的内容。

思路：为每个候选问题列出"若要回答它，库里就**必须**出现的特征词"，
逐个在 chunks 里做 LIKE 统计。全部为 0 才算是合格的负样本。

用法：python scripts/check_negative_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402

# (问题, [只有能回答它才可能出现的特征词])
CANDIDATES: list[tuple[str, list[str]]] = [
    ("厂房内疏散楼梯的最小净宽度是多少？", ["疏散楼梯"]),
    ("《石油化工企业设计防火规范》GB 50160 中甲B类液体储罐的防火间距是多少米？", ["甲B", "甲 B"]),
    ("《上海市消防条例》要求消防安全重点单位多久进行一次灭火演练？", ["上海市消防条例", "上海"]),
    ("自动喷水灭火系统的喷头应当多久更换一次？", ["喷头"]),
    ("2023 年全国共发生多少起火灾？", ["火灾起数", "万起"]),
    ("消防安全责任人每年应当接受多少学时的消防安全培训？", ["学时"]),
    ("《建筑灭火器配置设计规范》GB 50140 规定的灭火器配置基准是多少？", ["GB 50140", "配置基准", "灭火器配置设计"]),
    ("电动自行车在厂房内充电应当符合什么要求？", ["电动自行车", "电动车"]),
]


def main() -> int:
    with engine.connect() as c:
        total = c.execute(text("SELECT count(*) FROM chunks")).scalar()
        print(f"知识库共 {total} 块\n")
        print(f"{'特征词':<16}{'命中块数':>8}   问题")
        print("-" * 92)
        verdicts: list[tuple[str, bool]] = []
        for question, terms in CANDIDATES:
            counts = []
            for t in terms:
                n = c.execute(
                    text("SELECT count(*) FROM chunks WHERE content LIKE :k"), {"k": f"%{t}%"}
                ).scalar()
                counts.append((t, n))
            ok = all(n == 0 for _, n in counts)
            verdicts.append((question, ok))
            detail = "，".join(f"{t}={n}" for t, n in counts)
            print(f"{detail:<34} {question[:44]}")
            print(f"{'  → ' + ('✅ 合格（库中无相关内容）' if ok else '⚠️ 库中出现过，需人工确认是否能回答'):<40}")
        print("-" * 92)
        good = [q for q, ok in verdicts if ok]
        print(f"\n合格负样本 {len(good)} / {len(CANDIDATES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
