"""检索结果融合。

第一批用 LlamaIndex 内置的 QueryFusionRetriever：
- mode=reciprocal_rerank -> RRF 倒数排名融合，对分数尺度不敏感，**推荐**；
- mode=relative_score   -> 相对分数融合，需要两路分数可比。

RRF 公式： score(d) = Σ 1 / (k + rank_i(d))，k 默认 60。
它天然解决"向量相似度 0.83 与 BM25 得分 12.7 不可比"的问题。
"""

from __future__ import annotations

from typing import Any, Literal

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

FusionMode = Literal["reciprocal_rerank", "relative_score", "dist_based_score", "simple"]


def build_fusion_retriever(
    retrievers: list[Any],
    top_k: int | None = None,
    mode: FusionMode = "reciprocal_rerank",
    num_queries: int = 1,
    llm: Any | None = None,
) -> Any:
    """把多路检索器包成 QueryFusionRetriever。

    参数说明：
    - num_queries=1：不额外生成子查询，只做融合（省 LLM 调用）；
    - num_queries>1：自动做 Multi-Query 扩写（费用换召回），需要 llm 才能工作；
    - use_async=True：两路并行检索，延迟从相加变成取最大值。
    """
    from llama_index.core.retrievers import QueryFusionRetriever

    # 必须显式传 llm：QueryFusionRetriever 在 llm 为空时会回落到 Settings.llm，
    # 而 Settings.llm 的默认值是 OpenAI，会直接抛 "No API key found for OpenAI"。
    # 即使 num_queries=1（不做扩写）它也会在构造函数里解析一次 llm。
    from app.dependencies import ensure_llama_settings, llm_dep

    ensure_llama_settings()

    kwargs: dict[str, Any] = {
        "retrievers": retrievers,
        "similarity_top_k": top_k or settings.retrieve_top_k,
        "mode": mode,
        "num_queries": num_queries,
        # 见 config.retrieval_use_async 的说明：同步链路下必须关，否则第二次检索会崩
        "use_async": settings.retrieval_use_async,
        "llm": llm or llm_dep(),
        "verbose": settings.debug,
    }

    logger.info("构建融合检索器：mode=%s, 路数=%s, top_k=%s, use_async=%s",
                mode, len(retrievers), kwargs["similarity_top_k"], kwargs["use_async"])
    return QueryFusionRetriever(**kwargs)


def dedupe_nodes(nodes: list[Any]) -> list[Any]:
    """按 node_id 去重（融合后同一片段可能被多路召回）。"""
    seen: set[str] = set()
    result: list[Any] = []
    for node in nodes:
        node_id = getattr(node.node, "node_id", None) or getattr(node, "node_id", "")
        if node_id in seen:
            continue
        seen.add(node_id)
        result.append(node)
    return result


def nodes_to_sources(nodes: list[Any], preview_chars: int = 300) -> list[dict]:
    """把检索结果转成 API 用的 source 字典（前端展示出处）。

    TODO(第二批): 把 retrieval_source 标注成 vector/bm25/两者都命中，
                  便于在评估里判断"稀疏路贡献度"。
    """
    sources: list[dict] = []
    for n in nodes:
        node = getattr(n, "node", n)
        meta = node.metadata or {}
        sources.append(
            {
                "chunk_id": meta.get("chunk_id", node.node_id),
                "document_id": meta.get("document_id", ""),
                "document_title": meta.get("title", ""),
                "heading_path": meta.get("heading_path", ""),
                "page_no": int(meta.get("page_no", 0) or 0),
                "score": float(getattr(n, "score", 0.0) or 0.0),
                "content": node.get_content()[:preview_chars],
                "retrieval_source": meta.get("retrieval_source", ""),
            }
        )
    return sources
