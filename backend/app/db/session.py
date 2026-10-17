"""SQLAlchemy 引擎 / SessionLocal / FastAPI 依赖。

⚠️ Neon 注意事项：
1. **必须用 pooler 连接串**（主机名形如 `ep-xxx-pooler.xxx.aws.neon.tech`），
   直连串在 Serverless 冷启动时会耗尽连接；
2. pooler 默认是 PgBouncer transaction 模式，**不能用服务端 prepared statement**，
   因此 psycopg2 要关掉自动 prepare：见 `_engine_kwargs` 里的 connect_args；
3. 连接串务必带 `?sslmode=require`。
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import Base
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _is_local_host(url: str) -> bool:
    """判断是不是本地数据库（localhost / 127.0.0.1 / ::1）。"""
    from sqlalchemy.engine import make_url

    try:
        host = (make_url(url).host or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return host in {"localhost", "127.0.0.1", "::1", ""}


def _engine_kwargs() -> dict:
    """按数据库类型给出引擎参数（SQLite 只在测试里用）。"""
    if settings.sqlalchemy_url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}

    connect_args: dict = {}
    if "sslmode=" not in settings.database_url and not _is_local_host(settings.database_url):
        # 远程库（Neon 等）必须走 TLS；本地库不强制（本地 Postgres 常没开 SSL，
        # 硬加 require 会连不上，此时请在 DATABASE_URL 里显式写 ?sslmode=disable）
        connect_args["sslmode"] = "require"

    # ⚠️ Neon 的 pooler（PgBouncer）**不允许**通过 options 传启动参数，否则报：
    #    ERROR: unsupported startup parameter in options: statement_timeout
    # 因此只在直连串（无 -pooler）上设置语句超时；pooler 上改为每条事务内 SET LOCAL。
    if "-pooler" not in settings.database_url:
        connect_args["options"] = f"-c statement_timeout={settings.db_statement_timeout_ms}"

    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": 5,
        # Neon 会主动断空闲连接故需探活；本地库没有这个问题，省掉每次取连接的额外往返
        "pool_pre_ping": not _is_local_host(settings.database_url),
        "pool_recycle": 300,
        "connect_args": connect_args,
    }


def _safe_url() -> str:
    """未配置 DATABASE_URL 时给一个占位串，保证 import 阶段不炸（真正的连接在首次查询时才建）。"""
    if settings.sqlalchemy_url:
        return settings.sqlalchemy_url
    logger.warning("未配置 DATABASE_URL，使用占位连接串；数据库相关接口会在调用时报错")
    return "postgresql+psycopg2://placeholder:placeholder@localhost:5432/placeholder"


def _build_engine():
    """建引擎；数据库驱动缺失时降级为内存 SQLite，保证 import 与单测不被环境阻断。"""
    try:
        return create_engine(_safe_url(), echo=settings.db_echo, future=True, **_engine_kwargs())
    except ModuleNotFoundError as exc:      # 例如未安装 psycopg2-binary
        logger.error("PostgreSQL 驱动缺失（%s），降级为内存 SQLite；生产部署请安装 psycopg2-binary", exc)
        return create_engine("sqlite+pysqlite:///:memory:", echo=False, future=True)


engine = _build_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求一个事务性 Session，结束自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """脚本 / 后台任务用的上下文管理器，带提交回滚。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """开发期建表（生产用 Alembic）。同时确保 pgvector 扩展存在。"""
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as exc:  # 非 superuser 时可能没权限，Neon 默认已开启
            logger.warning("创建 vector 扩展失败（可忽略，若已存在）: %s", exc)
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表已就绪（Base.metadata.create_all）")


def ping_db() -> bool:
    """健康检查用：能执行 SELECT 1 即认为数据库可用。"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("数据库探活失败: %s", exc)
        return False
