"""按编号把勾选清单里的条目置为不保留（_keep=false），再重跑 build_golden_set.py 生效。

用法：python scripts/drop_golden_items.py 1 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("data/eval/retrieval_golden_preselect.jsonl")


def main() -> int:
    wanted = {int(x) for x in sys.argv[1:]}
    if not wanted:
        print("用法：python scripts/drop_golden_items.py <编号> [编号...]")
        return 1

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    hit = 0
    for r in rows:
        if r["_no"] in wanted and r.get("_keep"):
            r["_keep"] = False
            hit += 1
            print(f"  已置 _keep=false：[{r['_no']}] {r['question'][:52]}")
        elif r["_no"] in wanted:
            print(f"  （[{(r['_no'])}] 本来就是 false，跳过）")

    missing = wanted - {r["_no"] for r in rows}
    if missing:
        print(f"  ⚠️ 这些编号不存在：{sorted(missing)}")

    SRC.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"\n共修改 {hit} 条；现在保留 {sum(1 for r in rows if r.get('_keep'))} 条")
    print("接着运行：python scripts/build_golden_set.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
