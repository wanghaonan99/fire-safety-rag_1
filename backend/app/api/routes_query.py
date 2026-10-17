"""问答接口：POST /api/query（带会话记忆 + 语义缓存）。

一次完整问答的链路：
    语义缓存查 -> 指代消解(condense) -> 查询改写(step-back/HyDE)
    -> 向量+BM25 融合检索 -> Rerank 精排 -> 拼 prompt -> DeepSeek 生成
    -> 写会话记忆 -> 写语义缓存 -> 返回答案与出处

每一步都可单独关闭/替换，便于按评估结果做消融实验。
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.core.generation.prompt import build_condense_prompt, build_qa_messages
from app.core.memory.session_store import SessionStore
from app.core.retrieval.fusion import nodes_to_sources
from app.db.session import get_db
from app.models.request import QueryRequest
from app.models.response import ApiResponse, QueryResponse, SourceRef
from app.utils.cache import get_cached, set_cached
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["query"])


def _condense_question(question: str, history: list[dict[str, str]], llm) -> str:
    """多轮场景下把"它/这个"补全为独立问题。首轮或失败时原样返回。"""
    if not history:
        return question
    try:
        response = llm.complete(build_condense_prompt(question, history))
        condensed = str(getattr(response, "text", response)).strip().splitlines()[0]
        if condensed and len(condensed) < len(question) * 4:
            logger.info("指代消解: %s -> %s", question, condensed)
            return condensed
    except Exception as exc:  # noqa: BLE001
        logger.warning("指代消解失败，使用原问题: %s", exc)
    return question


def _retrieve(index, question: str, top_k: int, use_rerank: bool, doc_category: str | None = None):
    """跑检索链路，返回节点列表。

    用带缓存的 get_hybrid_retriever：每请求重建检索器（含 BM25 索引与融合器）
    实测要比复用多花约 8 秒。
    """
    from app.core.retrieval.hybrid import get_hybrid_retriever

    retriever = get_hybrid_retriever(
        index, use_rerank=use_rerank, top_k=top_k, doc_category=doc_category
    )
    return retriever.retrieve(question)


@router.post("/query", response_model=ApiResponse[QueryResponse], summary="消防知识问答")
def query(payload: QueryRequest, db: Session = Depends(get_db)) -> ApiResponse[QueryResponse]:
    start = time.perf_counter()
    store = SessionStore(db)
    session = store.get_or_create(payload.session_id, user_id=payload.user_id)
    top_k = payload.top_k or settings.rerank_top_k

    # ---------- 0. 语义缓存 ----------
    if payload.use_cache:
        cached = get_cached(payload.question, doc_category=payload.doc_category, top_k=top_k)
        if cached:
            # 用批量写入：跨国库上 add_message×2 要 6 条 SQL 往返，add_turn 只要 4 条
            store.add_turn(
                session.id,
                payload.question,
                cached["answer"],
                sources=cached.get("sources"),
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
            return ApiResponse(
                data=QueryResponse(
                    answer=cached["answer"],
                    session_id=session.id,
                    sources=[SourceRef(**s) for s in cached.get("sources", [])],
                    cached=True,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    model=settings.deepseek_model,
                ),
                message="命中语义缓存",
            )

    # ---------- 1. 记忆与指代消解 ----------
    from app.dependencies import llm_dep

    llm = llm_dep()
    history = store.get_history(session.id, max_turns=settings.session_max_turns)
    standalone = _condense_question(payload.question, history, llm)
    store.rename_from_first_question(session, payload.question)

    # ---------- 2. 查询改写（HyDE / Step-back，默认关）----------
    from app.core.retrieval.query_transform import rewrite_queries

    queries, _ = rewrite_queries(
        standalone, llm=llm,
        enable_transform=payload.use_query_transform if payload.use_query_transform is not None
        else settings.enable_query_transform,
    )

    # ---------- 3. 混合检索 + 精排 ----------
    try:
        from app.db.vector_store import build_vector_index

        index = build_vector_index()
        nodes = _retrieve(
            index, queries[0], top_k,
            use_rerank=settings.rerank_provider != "none",
            doc_category=payload.doc_category,
        )
        # 多查询场景：其余改写结果再检索一次并合并去重
        if len(queries) > 1:
            from app.core.retrieval.fusion import dedupe_nodes

            for q in queries[1:]:
                nodes = dedupe_nodes(
                    nodes + _retrieve(index, q, top_k, use_rerank=False, doc_category=payload.doc_category)
                )
            nodes = sorted(nodes, key=lambda n: getattr(n, "score", 0.0) or 0.0, reverse=True)[: settings.rerank_top_k]
    except Exception as exc:
        logger.exception("检索失败")
        raise HTTPException(500, f"检索失败: {exc}") from exc

    if not nodes:
        answer = "根据现有资料无法确定，建议查阅相关规范原文或咨询安全管理部门。"
        store.add_turn(session.id, payload.question, answer)
        return ApiResponse(data=QueryResponse(answer=answer, session_id=session.id), message="未检索到相关内容")

    # ---------- 4. 生成 ----------
    messages = build_qa_messages(standalone, nodes, history=history)
    try:
        from app.core.generation.llm import chat_direct

        answer = chat_direct(messages)
    except Exception as exc:
        logger.exception("生成失败")
        raise HTTPException(502, f"LLM 调用失败: {exc}") from exc

    # ---------- 5. 返回与落库 ----------
    sources = nodes_to_sources(nodes)
    latency_ms = int((time.perf_counter() - start) * 1000)
    store.add_turn(session.id, payload.question, answer, sources=sources, latency_ms=latency_ms)

    if payload.use_cache:
        set_cached(payload.question, answer, sources=sources, doc_category=payload.doc_category, top_k=top_k)

    # TODO(第二批): 把本次 trace（检索片段、耗时、token）上报 LangFuse
    return ApiResponse(
        data=QueryResponse(
            answer=answer,
            session_id=session.id,
            sources=[SourceRef(**s) for s in sources],
            transformed_queries=queries[1:],
            cached=False,
            latency_ms=latency_ms,
            model=settings.deepseek_model,
        )
    )


@router.get("/sessions", response_model=ApiResponse[list[dict]], summary="会话列表")
def list_sessions(
    user_id: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> ApiResponse[list[dict]]:
    """列出会话（供前端"会话列表"用），按最近更新排序。

    ⚠️ 刻意只发 2 条 SQL：先查会话，再**一次性**把所有会话的最后一条消息捞回来。
    逐会话查最后一条消息（N+1）在跨国数据库上会是 1 秒 × N 的灾难。
    """
    from app.db.models import ChatMessage, ChatSession

    query = db.query(ChatSession)
    if user_id:
        query = query.filter(ChatSession.user_id == user_id)
    sessions = query.order_by(ChatSession.updated_at.desc()).limit(limit).all()
    if not sessions:
        return ApiResponse(data=[])

    ids = [s.id for s in sessions]
    previews: dict[str, str] = {}
    counts: dict[str, int] = {}
    rows = (
        db.query(ChatMessage.session_id, ChatMessage.role, ChatMessage.content, ChatMessage.id)
        .filter(ChatMessage.session_id.in_(ids))
        .order_by(ChatMessage.id)
        .all()
    )
    for session_id, role, content, _mid in rows:
        counts[session_id] = counts.get(session_id, 0) + 1
        previews[session_id] = (content or "")[:60]

    return ApiResponse(
        data=[
            {
                "id": s.id,
                "title": s.title or "新会话",
                "user_id": s.user_id,
                "turn_count": s.turn_count or 0,
                "message_count": counts.get(s.id, 0),
                "preview": previews.get(s.id, ""),
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ]
    )


@router.get("/sessions/{session_id}/history", response_model=ApiResponse[list[dict]], summary="查看会话历史")
def get_history(session_id: str, db: Session = Depends(get_db)) -> ApiResponse[list[dict]]:
    store = SessionStore(db)
    return ApiResponse(data=store.get_history(session_id, max_turns=50))


@router.delete("/sessions/{session_id}", response_model=ApiResponse[dict], summary="清空会话")
def clear_session(session_id: str, db: Session = Depends(get_db)) -> ApiResponse[dict]:
    return ApiResponse(data={"deleted": SessionStore(db).clear(session_id)})
