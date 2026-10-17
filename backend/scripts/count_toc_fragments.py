"""统计库里残留的"目次碎片"切块（低信息密度过滤的漏网之鱼）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402


def main() -> int:
    with engine.connect() as c:
        total = c.execute(text("SELECT count(*) FROM chunks")).scalar()
        for pattern, label in [
            ("%....%", "含 4+ 连续点"),
            ("%........%", "含 8+ 连续点（强目次特征）"),
        ]:
            n = c.execute(
                text("SELECT count(*) FROM chunks WHERE content LIKE :p"), {"p": pattern}
            ).scalar()
            print(f"  {label:26} {n:>3} / {total} 块")

        print("\n强目次特征切块样例：")
        rows = c.execute(
            text(
                "SELECT d.title, k.content FROM chunks k JOIN documents d ON d.id = k.document_id "
                "WHERE k.content LIKE '%........%' LIMIT 5"
            )
        ).all()
        for title, content in rows:
            print(f"  《{title[:22]}》{content[:88].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
