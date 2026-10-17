"""验证 PGVectorStore 的同步 + 异步连接（不加载任何模型，几秒出结果）。

专门用来复现/回归这两类问题：
1. 只传 connection_string 时异步串被拼成 `...@None:None/None`；
2. Neon pooler 下异步侧需要 prepared_statement_cache_size=0。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.db.vector_store import VECTOR_TABLE_NAME, _build_connection_strings  # noqa: E402


def mask(cs: str) -> str:
    """打印连接串时隐藏密码。"""
    from sqlalchemy.engine import make_url

    u = make_url(cs)
    return f"{u.drivername}://{u.username}:***@{u.host}:{u.port}{u.database}?{u.query}"


def main() -> int:
    sync_cs, async_cs = _build_connection_strings()
    print("同步串:", mask(sync_cs))
    print("异步串:", mask(async_cs))
    print()

    # ---------- 同步 ----------
    from sqlalchemy import create_engine

    engine = create_engine(sync_cs, pool_pre_ping=True)
    with engine.connect() as conn:
        print("同步连接 ✓ SELECT 1 ->", conn.execute(text("SELECT 1")).scalar())
    engine.dispose()

    # ---------- 异步（原故障点）----------
    async def check_async() -> None:
        aengine = create_async_engine(async_cs, pool_pre_ping=True)
        async with aengine.connect() as conn:
            value = (await conn.execute(text("SELECT 1"))).scalar()
            print("异步连接 ✓ SELECT 1 ->", value)
        await aengine.dispose()

    asyncio.run(check_async())

    # ---------- 向量表是否存在 ----------
    with create_engine(sync_cs).connect() as conn:
        exists = conn.execute(
            text("SELECT to_regclass(:t)"), {"t": f"public.data_{VECTOR_TABLE_NAME}"}
        ).scalar()
        print(f"向量表 data_{VECTOR_TABLE_NAME}:", exists or "尚未创建（首次写入时自动建）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
