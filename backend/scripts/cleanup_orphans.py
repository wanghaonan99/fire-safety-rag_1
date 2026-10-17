"""清掉 chunks 表里已不存在的孤儿向量，并打印一致性检查结果。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402
from app.db.vector_store import VECTOR_TABLE_NAME, cleanup_orphan_nodes  # noqa: E402


def report(label: str) -> None:
    with engine.connect() as c:
        chunks = c.execute(text("SELECT count(*) FROM chunks")).scalar()
        vecs = c.execute(text(f"SELECT count(*) FROM data_{VECTOR_TABLE_NAME}")).scalar()
        print(f"  {label}: 切块 {chunks} · 向量 {vecs} · {'✅ 一致' if chunks == vecs else '❌ 不一致'}")


def main() -> int:
    report("清理前")
    n = cleanup_orphan_nodes()
    print(f"  已清理孤儿向量 {n} 条")
    report("清理后")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
