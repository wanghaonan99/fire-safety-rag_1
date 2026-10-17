"""打印某个目次候选切块的完整内容与点线统计，用于判断该不该删。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402


def main() -> int:
    with engine.connect() as c:
        rows = c.execute(
            text(
                "SELECT k.id, k.heading_path, k.content FROM chunks k "
                "WHERE k.content LIKE '%附录 A 防火间距起止点%' LIMIT 3"
            )
        ).all()
        for cid, heading, content in rows:
            dot_runs = re.findall(r"\.{4,}", content)
            dot_chars = sum(len(r) for r in dot_runs)
            compact = len(re.sub(r"\s", "", content)) or 1
            print("=" * 88)
            print(f"id={cid}")
            print(f"标题：{heading[:60]}")
            print(f"长度 {len(content)} · 点线字符 {dot_chars} · 占比 {dot_chars / compact:.2f}")
            print("完整内容：")
            print(content[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
