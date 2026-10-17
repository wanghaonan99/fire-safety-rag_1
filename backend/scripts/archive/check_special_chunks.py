"""检查"表格/技术图"是否真的进了库，并抽样展示。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402


def main() -> int:
    with engine.connect() as c:
        print("=== 特殊内容统计（共 %s 块）===" % c.execute(text("SELECT count(*) FROM chunks")).scalar())
        checks = [
            ("【表格", "含 Markdown 表格"),
            ("| ---", "含 Markdown 表格分隔行"),
            ("【本页含", "含技术图标记"),
            ("【第", "含页码标记"),
        ]
        for kw, label in checks:
            n = c.execute(
                text("SELECT count(*) FROM chunks WHERE content LIKE :k"), {"k": f"%{kw}%"}
            ).scalar()
            print(f"  {label:22} {n:>4} 块")

        print("\n=== 样例 1：防火间距表切块 ===")
        row = c.execute(
            text(
                "SELECT d.title, k.heading_path, k.content FROM chunks k "
                "JOIN documents d ON d.id = k.document_id "
                "WHERE k.content LIKE '%居住区、村镇及重要公共建筑%' AND k.content LIKE '%| 90 |%' LIMIT 1"
            )
        ).first()
        if row:
            print(f"  来源《{row[0]}》 {row[1][:40]}")
            print("  " + row[2][:400].replace("\n", "\n  "))
        else:
            print("  ⚠️ 没找到（表格可能被切块拆开了）")

        print("\n=== 样例 2：技术图标记所在的切块 ===")
        row = c.execute(
            text(
                "SELECT d.title, k.heading_path, k.content FROM chunks k "
                "JOIN documents d ON d.id = k.document_id "
                "WHERE k.content LIKE '%【本页含%' LIMIT 1"
            )
        ).first()
        if row:
            print(f"  来源《{row[0]}》 {row[1][:40]}")
            print("  " + row[2][:300].replace("\n", "\n  "))
        else:
            print("  ⚠️ 没找到")

        print("\n=== 样例 3：消防法切块（验证双栏修正 + 全角归一化）===")
        row = c.execute(
            text(
                "SELECT k.heading_path, k.content FROM chunks k "
                "JOIN documents d ON d.id = k.document_id "
                "WHERE d.title LIKE '中华人民共和国消防法%' LIMIT 1"
            )
        ).first()
        if row:
            print(f"  {row[0][:40]}")
            print("  " + row[1][:260].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
