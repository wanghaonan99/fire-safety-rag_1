"""Rerank 精排：BAAI/bge-reranker-v2-m3 交叉编码器。

为什么需要：
向量检索是双塔（query/doc 分别编码），快但粗；reranker 是交叉编码器，
把 (query, doc) 拼一起过模型，精度高但慢。所以只在**召回后 top 20-50** 上跑。

代价提示：本地 reranker 约占 2.3GB 内存，Render 免费套餐（512MB）跑不动，
线上把 RERANK_PROVIDER 设为 none，或换 API 版 rerank。
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_reranker() -> Any | None:
    """返回 LlamaIndex 的 reranker 对象；不可用时返回 None（检索链路自动跳过）。"""
    if settings.rerank_provider == "none":
        logger.info("RERANK_PROVIDER=none，跳过精排")
        return None

    try:
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

        # 同 embedding：优先用本地完整快照，避免断网时 transformers 的 Hub 查询踩坑
        from app.core.ingestion.embedder import find_cached_model_path

        local_path = find_cached_model_path(settings.rerank_model)
        model_ref = local_path or settings.rerank_model

        # FlagEmbeddingReranker 没有 device 参数，它内部自动选 CUDA；fp16 只在 GPU 上开
        gpu = _cuda_available()
        use_fp16 = bool(settings.rerank_use_fp16 or gpu)
        logger.info(
            "加载本地 Reranker：%s（device=%s, fp16=%s）",
            "本地快照" if local_path else settings.rerank_model + "（将联网下载）",
            "cuda" if gpu else "cpu",
            use_fp16,
        )
        return FlagEmbeddingReranker(
            model=model_ref,
            top_n=settings.rerank_top_k,
            use_fp16=use_fp16,
        )
    except Exception as exc:
        logger.warning("Reranker 加载失败，降级为纯融合排序: %s", exc)
        return None


def _cuda_available() -> bool:
    """判断 torch 是否真的能用 CUDA（CPU 版 torch 会返回 False）。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def wrap_with_rerank(retriever: Any, reranker: Any, top_n: int | None = None) -> Any:
    """把 retriever 包一层，使其 retrieve() 后自动精排。

    做法与 LlamaIndex 的 RetrieverQueryEngine 一致，但保持"可单独调用 retriever"的形态，
    方便评估脚本直接调 retriever 拿指标。
    """
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle

    original_retrieve = retriever.retrieve

    def _retrieve_with_rerank(query: Any, *args, **kwargs) -> list[NodeWithScore]:
        query_bundle = query if isinstance(query, QueryBundle) else QueryBundle(query_str=str(query))
        nodes = original_retrieve(query_bundle, *args, **kwargs)
        try:
            return reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
        except Exception as exc:
            logger.warning("精排失败，返回融合结果: %s", exc)
            return nodes

    retriever.retrieve = _retrieve_with_rerank

    async def _aretrieve_with_rerank(query: Any, *args, **kwargs) -> list[NodeWithScore]:
        if hasattr(retriever, "_aretrieve"):
            query_bundle = query if isinstance(query, QueryBundle) else QueryBundle(query_str=str(query))
            nodes = await retriever._aretrieve(query_bundle, *args, **kwargs)
        else:
            nodes = original_retrieve(query)
        return reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=str(query)))

    if isinstance(retriever, BaseRetriever) and hasattr(retriever, "_aretrieve"):
        retriever._aretrieve = _aretrieve_with_rerank

    retriever._reranker = reranker
    retriever._rerank_top_n = top_n or settings.rerank_top_k
    return retriever


def rerank_nodes(query: str, nodes: list[Any], top_n: int | None = None) -> list[Any]:
    """函数式精排入口，评估脚本/自定义链路直接用。"""
    from app.dependencies import reranker_dep

    reranker = reranker_dep()      # 单例：新建实例等于重新加载 2.3GB 模型
    if reranker is None:
        return nodes[: top_n or settings.rerank_top_k]
    from llama_index.core.schema import QueryBundle

    return reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
