"""管理接口：系统状态、缓存管理、配置查看。

第一批只做单口令鉴权（口令从环境变量 ADMIN_TOKEN 读取）；
上生产前建议换成真实鉴权（FastAPI OAuth2 / Clerk / Auth0）并加 IP 白名单。
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ChatSession, Chunk, Document, EvalRecord
from app.db.session import get_db
from app.models.request import LoginRequest
from app.models.response import ApiResponse
from app.utils.cache import clear_cache
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """管理鉴权：请求头 X-Admin-Token 必须与环境变量 ADMIN_TOKEN 一致。

    用 hmac.compare_digest 做常时间比较，避免通过响应时间猜测口令。
    """
    expected = settings.admin_token
    if expected == "change-me":
        logger.warning("ADMIN_TOKEN 仍是默认值 change-me，上生产前务必在 .env 中修改")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "管理口令无效（请在请求头带 X-Admin-Token）")


@router.post("/login", response_model=ApiResponse[dict], summary="管理登录")
def login(payload: LoginRequest) -> ApiResponse[dict]:
    if not hmac.compare_digest(payload.password, settings.admin_token):
        raise HTTPException(401, "口令错误")
    return ApiResponse(data={"token": settings.admin_token})


@router.get("/stats", response_model=ApiResponse[dict], summary="知识库统计")
def stats(db: Session = Depends(get_db), _: None = Depends(require_admin)) -> ApiResponse[dict]:
    """文档数 / 切块数 / 会话数 / 评估次数 / 各类别分布。"""
    by_category = dict(
        db.query(Document.doc_category, func.count(Document.id)).group_by(Document.doc_category).all()
    )
    by_status = dict(db.query(Document.status, func.count(Document.id)).group_by(Document.status).all())
    return ApiResponse(
        data={
            "documents": db.query(func.count(Document.id)).scalar() or 0,
            "chunks": db.query(func.count(Chunk.id)).scalar() or 0,
            "sessions": db.query(func.count(ChatSession.id)).scalar() or 0,
            "eval_records": db.query(func.count(EvalRecord.id)).scalar() or 0,
            "by_category": by_category,
            "by_status": by_status,
            "scanned_docs": db.query(func.count(Document.id)).filter(Document.is_scanned.is_(True)).scalar() or 0,
        }
    )


@router.get("/config", response_model=ApiResponse[dict], summary="查看生效配置（脱敏）")
def show_config(_: None = Depends(require_admin)) -> ApiResponse[dict]:
    """排查"为什么线上效果和本地不一样"的第一手段：把关键参数打出来。"""
    return ApiResponse(
        data={
            "environment": settings.environment,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "rerank_provider": settings.rerank_provider,
            "rerank_model": settings.rerank_model,
            "deepseek_model": settings.deepseek_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieve_top_k": settings.retrieve_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "enable_hype": settings.enable_hype,
            "enable_hyde": settings.enable_hyde,
            "enable_query_transform": settings.enable_query_transform,
            "semantic_cache_enabled": settings.semantic_cache_enabled,
            "redis_configured": bool(settings.redis_url),
            "langfuse_enabled": settings.langfuse_enabled,
            "db_configured": bool(settings.database_url),
        }
    )


@router.post("/cache/clear", response_model=ApiResponse[dict], summary="清空语义缓存")
def clear_semantic_cache(_: None = Depends(require_admin)) -> ApiResponse[dict]:
    return ApiResponse(data={"removed": clear_cache()}, message="缓存已清空")


# TODO(第二批):
#   POST /api/admin/warmup   预热 embedding / reranker（避免首个请求超时）
#   POST /api/admin/traces   查询 LangFuse 最近 trace，定位慢请求与幻觉样本
