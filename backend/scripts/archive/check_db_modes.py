"""校验不同 DATABASE_URL 下生成的连接参数（线上 Neon vs 本地 Postgres）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.engine import make_url

from app.config import settings
from app.db import session as s
from app.db.vector_store import _build_connection_strings

CASES = [
    ("线上 Neon pooler", "postgresql://u:p@ep-x-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"),
    ("本地 Docker 默认", "postgresql://postgres:postgres@localhost:5432/postgres"),
    ("本地显式关 SSL", "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"),
]

original = settings.database_url
try:
    for label, url in CASES:
        settings.database_url = url
        _, async_cs = _build_connection_strings()
        kwargs = s._engine_kwargs()
        ssl_async = make_url(async_cs).query.get("ssl", "(不设)")
        print(f"{label}")
        print(f"  同步 connect_args = {kwargs['connect_args']}")
        print(f"  同步 pool_pre_ping = {kwargs['pool_pre_ping']}")
        print(f"  异步 ssl          = {ssl_async}")
        print()
finally:
    settings.database_url = original
    print("已还原为 .env 里的线上连接串")
