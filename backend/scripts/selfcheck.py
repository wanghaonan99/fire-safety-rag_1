"""环境自检（可重复运行，不改动任何配置）。

用法：cd backend && python scripts/selfcheck.py
检查项：pooler/直连的 connect_args 分支、.env 关键项（脱敏）、真实连一次 Neon。
不需要了直接删掉本文件即可。
"""

from __future__ import annotations

from urllib.parse import urlsplit

from app.config import settings
from app.db import session as s


def mask(v: str) -> str:
    return "(未填)" if not v else f"已填({len(v)}字符)"


print("=== 1. connect_args 分支逻辑 ===")
_original = settings.database_url          # 先备份，最后还原，避免污染后面的输出
try:
    for label, url in [
        ("pooler（你的场景）", "postgresql://u:p@ep-x-pooler.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"),
        ("直连 unpooled", "postgresql://u:p@ep-x.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"),
    ]:
        settings.database_url = url
        args = s._engine_kwargs()["connect_args"]
        print(f"  {label:16} -> {args}")
        assert "-pooler" not in url or "options" not in args, "pooler 不应带 options！"
finally:
    settings.database_url = _original
print("  ✓ pooler 不再下发 options 启动参数")

print()
print("=== 2. .env 关键项（脱敏）===")
parts = urlsplit(settings.database_url)
print("  DEEPSEEK_API_KEY    :", mask(settings.deepseek_api_key))
print("  DATABASE_URL host   :", parts.hostname)
print("    -pooler:", "-pooler" in (parts.hostname or ""), "| sslmode:", "sslmode=" in (parts.query or ""),
      "| len:", len(settings.database_url))
print("  EMBEDDING_PROVIDER  :", settings.embedding_provider, "| model:", settings.embedding_model, "| dim:", settings.embedding_dim)
print("  RERANK_PROVIDER     :", settings.rerank_provider, "| model:", settings.rerank_model)
print("  ADMIN_TOKEN         :", "⚠ 仍是模板里的说明文字，需要换成真口令" if len(settings.admin_token) > 40 else "已自定义")
print("  CORS_ORIGINS        :", settings.cors_origins)

print()
print("=== 3. 实际连一次 PostgreSQL ===")
try:
    from sqlalchemy import text

    with s.engine.begin() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
        print("  ✓ 连接成功：", str(version)[:60])
        has_vector = conn.execute(
            text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        print("  pgvector 扩展:", "已启用" if has_vector else "❌ 未启用（请在 Neon SQL Editor 执行 CREATE EXTENSION IF NOT EXISTS vector;）")
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        ).scalars().all()
        print("  已有表:", tables or "(空，首次建表中)")
except Exception as exc:
    print("  ❌ 连接失败:", str(exc)[:300])
