"""LlamaIndex PGVectorStore 构建（向量库 = Neon PostgreSQL + pgvector）。

关键点：
1. 连接串从环境变量 DATABASE_URL 读取（**必须用 Neon 的 -pooler 串**）；
2. `table_name="fire_chunks"` 会自建表 `data_fire_chunks`，无需手工建表；
3. `embed_dim` 必须与 EMBEDDING_DIM 一致，否则插入报维度错误；
4. 换 embedding 模型时：改维度 + DROP TABLE data_fire_chunks 重新导入。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sqlalchemy.engine import make_url

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

VECTOR_TABLE_NAME = "fire_chunks"


def _build_connection_strings() -> tuple[str, str]:
    """把 DATABASE_URL 拆成同步（psycopg2）与异步（asyncpg）两个连接串。

    为什么必须显式给两个串：
    PGVectorStore.from_params 里是这么写的——
        conn_str       = connection_string or f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        async_conn_str = async_connection_string or f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    只传 connection_string 时，异步串会用**全为 None 的组件**拼成
    `postgresql+asyncpg://None:None@None:None/None`，随后 create_async_engine 报
    `ValueError: invalid literal for int() with base 10: 'None'`。

    另外两个 Neon 专属适配：
    - asyncpg 不认识 libpq 的 `sslmode` / `channel_binding` 参数，它们会被原样透传给
      asyncpg.connect() 而报 TypeError；异步串改用 asyncpg 自己的 `ssl=require`。
    - pooler（PgBouncer transaction 模式）不支持服务端 prepared statement，异步侧必须
      设 `prepared_statement_cache_size=0`，否则会随机报
      `prepared statement "__asyncpg_stmt_x__" already exists`。
    """
    url = make_url(settings.database_url)

    async_url = url.set(drivername="postgresql+asyncpg")
    query = dict(async_url.query)
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)

    # SSL 策略：显式写了 sslmode 就照做；没写时，远程库要求 TLS，本地库不强制。
    # （本地 Postgres 常常没开 SSL，硬加 require 会直接连不上）
    host = (async_url.host or "").lower()
    if sslmode:
        query.setdefault("ssl", "disable" if str(sslmode).lower() == "disable" else "require")
    elif host not in {"localhost", "127.0.0.1", "::1", ""}:
        query.setdefault("ssl", "require")

    query["prepared_statement_cache_size"] = "0"
    async_url = async_url.set(query=query)

    # render_as_string(hide_password=False)：URL 的 str() 会把密码变成 ***
    return (
        url.render_as_string(hide_password=False),
        async_url.render_as_string(hide_password=False),
    )


def build_vector_store(embed_model: Any | None = None) -> Any:
    """返回配置好的 PGVectorStore。"""
    from llama_index.vector_stores.postgres import PGVectorStore

    if not settings.database_url:
        raise ValueError("未配置 DATABASE_URL，无法连接 Neon PostgreSQL")

    sync_cs, async_cs = _build_connection_strings()
    logger.info("连接 pgvector（table=%s, dim=%s）", VECTOR_TABLE_NAME, settings.embedding_dim)
    return PGVectorStore.from_params(
        connection_string=sync_cs,
        async_connection_string=async_cs,
        table_name=VECTOR_TABLE_NAME,
        embed_dim=settings.embedding_dim,
        # HNSW 建索引更快、查询更稳；相比 IVFFlat 不需要预训练数据
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
        # 交给 LlamaIndex 的异步/多进程控制；两个引擎都接受这两个参数
        perform_setup=True,
        use_jsonb=True,
        create_engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300},
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Any:
    """进程级单例，避免每次请求都建连接池。"""
    return build_vector_store()


_INDEX_SINGLETON: Any = None
_INDEX_SINGLETON_EMBED_ID: int | None = None


def invalidate_index_cache() -> None:
    """知识库变化或换 embedding 模型后调用。"""
    global _INDEX_SINGLETON, _INDEX_SINGLETON_EMBED_ID
    _INDEX_SINGLETON = None
    _INDEX_SINGLETON_EMBED_ID = None


def build_vector_index(nodes: list | None = None, embed_model: Any | None = None) -> Any:
    """用 PGVectorStore 建/取 VectorStoreIndex。

    不传 nodes 时返回**进程内单例**：跨区域数据库上，每个请求重新
    `from_vector_store()` 都会触发一次连接与元数据读取，实测 0.6~2 秒。
    """
    global _INDEX_SINGLETON, _INDEX_SINGLETON_EMBED_ID
    from llama_index.core import StorageContext, VectorStoreIndex

    from app.dependencies import embed_model_dep

    embed = embed_model or embed_model_dep()
    storage_context = StorageContext.from_defaults(vector_store=get_vector_store())

    if nodes:
        return VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed)

    if _INDEX_SINGLETON is None or _INDEX_SINGLETON_EMBED_ID != id(embed):
        _INDEX_SINGLETON = VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=embed)
        _INDEX_SINGLETON_EMBED_ID = id(embed)
    return _INDEX_SINGLETON


def delete_document_nodes(document_id: str) -> int:
    """删除某文档的全部向量，返回删除行数。

    ⚠️ 不要用 `metadata_->>'document_id'` 作为主要条件：实测该版本的 PGVectorStore
    写进去的 metadata_ 顶层**没有** document_id（取出来是 NULL），按它删会一行都删不掉，
    于是"重建索引"变成"向量越堆越多"（一次 64 块变成 128 行）。

    可靠做法：用本项目 chunks 表里的 id —— 它就是写入向量时的 node_id。
    """
    from sqlalchemy import bindparam, text

    from app.db.session import engine

    deleted = 0
    with engine.begin() as conn:
        ids = (
            conn.execute(text("SELECT id FROM chunks WHERE document_id = :doc"), {"doc": document_id})
            .scalars()
            .all()
        )
        if ids:
            stmt = text(f"DELETE FROM data_{VECTOR_TABLE_NAME} WHERE node_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            )
            deleted += conn.execute(stmt, {"ids": list(ids)}).rowcount or 0

        # 兜底：万一将来 LlamaIndex 版本把 document_id 提到了顶层，也一并清掉
        try:
            result = conn.execute(
                text(f"DELETE FROM data_{VECTOR_TABLE_NAME} WHERE metadata_->>'document_id' = :doc"),
                {"doc": document_id},
            )
            deleted += result.rowcount or 0
        except Exception:  # noqa: BLE001  列名/结构变化时忽略
            pass
    return deleted


def cleanup_orphan_nodes() -> int:
    """删除 chunks 表里已不存在的向量行（失败重试、强制重建后的残留）。

    "向量行数 ≠ 切块数"时调用它，或直接用 ingest_cli 的 --reset 全量重建。
    """
    from sqlalchemy import text

    from app.db.session import engine

    with engine.begin() as conn:
        result = conn.execute(
            text(
                f"DELETE FROM data_{VECTOR_TABLE_NAME} "
                "WHERE node_id NOT IN (SELECT id FROM chunks)"
            )
        )
        orphans = result.rowcount or 0
    if orphans:
        logger.warning("清理了 %s 条孤儿向量（chunks 表中已不存在）", orphans)
    return orphans


def drop_vector_table() -> None:
    """危险操作：清空向量表（换 embedding 维度或全量重建时用）。"""
    from sqlalchemy import text

    from app.db.session import engine

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS data_{VECTOR_TABLE_NAME}"))
    get_vector_store.cache_clear()
    logger.warning("已删除向量表 data_%s", VECTOR_TABLE_NAME)
