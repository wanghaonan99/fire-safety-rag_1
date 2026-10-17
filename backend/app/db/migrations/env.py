"""Alembic 环境配置：连接串从 app.config.settings 读取（即 .env 的 DATABASE_URL）。

⚠️ 迁移时建议用 Neon 的**直连串**（非 -pooler），因为 DDL 在 PgBouncer
   transaction 模式下可能不稳定；迁移完再切回 pooler 跑服务。
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 用运行时配置覆盖 alembic.ini 里的空值
config.set_main_option("sqlalchemy.url", settings.sqlalchemy_url)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """排除 LlamaIndex 自建表与 pgvector 扩展对象，避免 autogenerate 生成删除语句。"""
    if type_ == "table" and name and (name.startswith("data_") or name == "spatial_ref_sys"):
        return False
    return True


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连库。"""
    context.configure(
        url=settings.sqlalchemy_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：直连数据库执行迁移。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # pgvector 扩展必须先存在，否则 vector 列类型迁移会失败
        connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
