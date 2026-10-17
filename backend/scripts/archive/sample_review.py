"""从勾选清单里抽样展示，便于人工快速判断草稿质量。"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path("data/eval/retrieval_golden_preselect.jsonl")


def main() -> int:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(7)

    clean = [r for r in rows if r["_keep"]]
    flagged = [r for r in rows if not r["_keep"]]

    print(f"共 {len(rows)} 条：无标记 {len(clean)}，需核对 {len(flagged)}\n")

    by_doc: dict[str, list] = defaultdict(list)
    for r in clean:
        by_doc[r["_doc"]].append(r)

    for doc, items in by_doc.items():
        print("=" * 84)
        print(f"【无标记 · 来源《{doc[:34]}》 共 {len(items)} 条】")
        for r in random.sample(items, min(3, len(items))):
            print(f"\n  [{r['_no']}] {r['type'] or '未标注'} · 第{r['_no']}条")
            print(f"      Q: {r['question']}")
            print(f"      A: {r['ground_truth'][:130]}")

    print("\n" + "=" * 84)
    print("【需核对 · 抽样（看标记是否判得对）】")
    for r in random.sample(flagged, min(4, len(flagged))):
        print(f"\n  [{r['_no']}] {'；'.join(r['_flags'])}")
        print(f"      Q: {r['question']}")
        print(f"      A: {r['ground_truth'][:110]}")

    # 事实型堆里有没有可以合并成归纳型的线索
    print("\n" + "=" * 84)
    print("【题型偏斜提示】")
    dist: dict[str, int] = defaultdict(int)
    for r in clean:
        dist[r["type"] or "未标注"] += 1
    for t, n in dist.items():
        print(f"  {t}: {n} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
