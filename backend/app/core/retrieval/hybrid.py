"""混合检索：BM25 稀疏检索 + 向量稠密检索，再用融合器合并。

为什么需要 BM25？
消防场景里大量精确术语（"GB 50016"、"甲类厂房"、"消火栓 15m"），
纯向量检索对这类"精确串 + 罕见词"召回很差，BM25 正好互补。

中文注意：BM25 依赖分词。这里用**字符 n-gram** 兜底（无需 jieba 词典，
对中文短语/编号效果可接受）；装了 jieba 时自动切换成词级分词。
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_BM25_CORPUS: list[Any] = []      # 进程内缓存节点，避免每次查询都回库拉全文
_RETRIEVER_CACHE: dict[tuple, Any] = {}   # 检索器单例：每请求重建一次要多花 ~8 秒


def invalidate_retriever_cache() -> None:
    """知识库变化后必须调用（导入/删除/重建索引）。"""
    _RETRIEVER_CACHE.clear()
    _BM25_CORPUS.clear()
    logger.info("已清空检索器缓存与 BM25 语料")


def get_hybrid_retriever(
    index: Any,
    use_rerank: bool | None = None,
    top_k: int | None = None,
    doc_category: str | None = None,
) -> Any:
    """带缓存的检索器工厂——**路由里必须用它**，不要每个请求都 build_hybrid_retriever()。

    实测（64 块语料，Neon 在新加坡）：
        每请求重建：31.5s（含首次语料加载）→ 稳态 11.2s / 11.6s
        复用同一个：9.96s（首次）→ 稳态 3.5s / 3.4s
    差在 BM25 索引与融合器（含分词器）的重复构建。

    缓存键包含 doc_category：不同类别各缓存一份（最多几个类别，代价可控）。
    """
    key = (bool(use_rerank), top_k, doc_category or "")
    if key not in _RETRIEVER_CACHE:
        logger.info(
            "构建并缓存检索器：use_rerank=%s top_k=%s doc_category=%s",
            use_rerank, top_k, doc_category or "(全部)",
        )
        _RETRIEVER_CACHE[key] = build_hybrid_retriever(
            index, use_rerank=use_rerank, top_k=top_k, doc_category=doc_category
        )
    return _RETRIEVER_CACHE[key]


def chinese_tokenizer(text: str) -> list[str]:
    """中文分词器：优先 jieba，缺失则用 2-gram 字符切分。"""
    try:
        import jieba  # type: ignore

        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        # 2-gram 兜底：把"灭火器配置"切成 灭火/火器/器配/配置
        cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
        tokens: list[str] = []
        for seg in cleaned.split():
            if seg.isascii():
                tokens.append(seg.lower())
            else:
                tokens.extend(seg[i : i + 2] for i in range(max(len(seg) - 1, 1)))
        return tokens


def build_bm25_retriever(nodes: list[Any] | None = None, top_k: int | None = None) -> Any:
    """构建 BM25 检索器。

    nodes 为空时从向量库拉取全部节点构建语料（首次调用稍慢，之后走缓存）。
    显式传入 nodes（例如按资料类别筛选后的子集）时**不覆盖全局语料缓存**，
    否则一次类别筛选会把全量语料污染成子集。
    """
    from llama_index.retrievers.bm25 import BM25Retriever

    global _BM25_CORPUS
    if nodes is None:
        if not _BM25_CORPUS:
            _BM25_CORPUS = _load_nodes_from_store()
        corpus = _BM25_CORPUS
    else:
        corpus = list(nodes)

    logger.info("BM25 语料节点数：%s", len(corpus))
    return BM25Retriever.from_defaults(
        nodes=corpus,
        similarity_top_k=top_k or settings.bm25_top_k,
        tokenizer=chinese_tokenizer,
    )


def _load_nodes_from_store() -> list[Any]:
    """从 pgvector 表里拉取全部节点，供 BM25 建语料。

    为什么不调 `store.get_nodes()`：PGVectorStore 没有这个方法，原实现返回空列表，
    会让 BM25Retriever.from_defaults(nodes=[]) 直接报错——即整条混合检索链路崩掉。
    这里直接查表（node_id / text / metadata_）。
    TODO(第二批): 语料量大时改为分页 + 磁盘缓存，避免每次启动都全表扫描。
    """
    from llama_index.core.schema import TextNode
    from sqlalchemy import text as sql_text

    from app.db.session import engine
    from app.db.vector_store import VECTOR_TABLE_NAME

    nodes: list[TextNode] = []
    sql = sql_text(f"SELECT node_id, text, metadata_ FROM data_{VECTOR_TABLE_NAME}")
    with engine.connect() as conn:
        for node_id, content, meta in conn.execute(sql).all():
            nodes.append(
                TextNode(id_=node_id, text=content or "", metadata=dict(meta or {}))
            )
    logger.info("从向量库加载 BM25 语料：%s 个节点", len(nodes))
    return nodes


def corpus_for_category(doc_category: str | None) -> list[Any]:
    """按资料类别筛选 BM25 语料（doc_category 为空则返回全量）。"""
    global _BM25_CORPUS
    if not _BM25_CORPUS:
        _BM25_CORPUS = _load_nodes_from_store()
    if not doc_category:
        return _BM25_CORPUS
    return [n for n in _BM25_CORPUS if (n.metadata or {}).get("doc_category") == doc_category]


def build_vector_retriever(
    index: Any, top_k: int | None = None, doc_category: str | None = None
) -> Any:
    """向量检索器（dense），可选按资料类别过滤。

    用 PGVectorStore 原生的 metadata filter，让过滤发生在**数据库端**，
    而不是取回 top_k 之后再丢——后者会白白浪费召回名额。
    """
    kwargs: dict[str, Any] = {"similarity_top_k": top_k or settings.similarity_top_k}
    if doc_category:
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

        kwargs["filters"] = MetadataFilters(
            filters=[MetadataFilter(key="doc_category", value=doc_category)]
        )
    return index.as_retriever(**kwargs)


def build_hybrid_retriever(
    index: Any,
    nodes: list[Any] | None = None,
    use_rerank: bool | None = None,
    top_k: int | None = None,
    doc_category: str | None = None,
) -> Any:
    """组装最终检索器：向量 + BM25 -> 融合 -> （可选）Rerank。

    返回的 retriever 上挂 `_pipeline_info`，便于 /api/eval 记录配置快照。
    """
    from app.core.retrieval.fusion import build_fusion_retriever
    from app.core.retrieval.reranker import wrap_with_rerank

    vector_retriever = build_vector_retriever(
        index, top_k=settings.retrieve_top_k, doc_category=doc_category
    )
    bm25_nodes = nodes if nodes is not None else corpus_for_category(doc_category)
    bm25_retriever = build_bm25_retriever(nodes=bm25_nodes, top_k=settings.bm25_top_k)

    retriever = build_fusion_retriever(
        retrievers=[vector_retriever, bm25_retriever],
        top_k=top_k or settings.retrieve_top_k,
    )
    retriever._pipeline_info = {"fusion": "QueryFusionRetriever", "rerank": False}

    enable_rerank = settings.rerank_provider != "none" if use_rerank is None else use_rerank
    if enable_rerank:
        # 用单例：build_reranker() 每次都新建实例 = 重新加载 2.3GB 模型（实测 5.7s）
        from app.dependencies import reranker_dep

        reranker = reranker_dep()
        if reranker is not None:
            retriever = wrap_with_rerank(retriever, reranker, top_n=settings.rerank_top_k)
            retriever._pipeline_info = {"fusion": "QueryFusionRetriever", "rerank": settings.rerank_model}

    return retriever
