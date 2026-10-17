"""打印候选负样本特征词在库中的命中上下文，判断是"提到"还是"能回答"。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402

TERMS = sys.argv[1:] or ["疏散楼梯", "甲 B", "上海", "GB 50140"]


def main() -> int:
    with engine.connect() as c:
        for term in TERMS:
            rows = c.execute(
                text(
                    "SELECT d.title, k.heading_path, k.content FROM chunks k "
                    "JOIN documents d ON d.id = k.document_id "
                    "WHERE k.content LIKE :k LIMIT 4"
                ),
                {"k": f"%{term}%"},
            ).all()
            print("=" * 88)
            print(f"【{term}】命中 {len(rows)} 块（最多显示 4 条）")
            for title, heading, content in rows:
                pos = content.find(term)
                lo = max(pos - 90, 0)
                snippet = content[lo : pos + 110].replace("\n", " ")
                print(f"  《{title[:24]}》{heading[:22]}")
                print(f"     …{snippet}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
