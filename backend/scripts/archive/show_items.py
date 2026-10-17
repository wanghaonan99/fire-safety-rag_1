"""打印指定编号的条目（含原文对照），用于人工复核。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402

WANTED = [int(x) for x in sys.argv[1:]] or [1, 4]


def main() -> int:
    rows = [
        json.loads(l)
        for l in Path("data/eval/retrieval_golden_preselect.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    by_no = {r["_no"]: r for r in rows}
    with engine.connect() as c:
        for no in WANTED:
            r = by_no.get(no)
            if not r:
                continue
            content = c.execute(
                text("SELECT content FROM chunks WHERE id = :i"), {"i": r["ground_truth_ids"][0]}
            ).scalar() or ""
            print("=" * 86)
            print(f"[{no}] {r['question']}")
            print(f"  标记：{'；'.join(r['_flags']) or '无'}")
            print(f"  答案：{r['ground_truth'][:200]}")
            print(f"  原文：{content[:300].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
