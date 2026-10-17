"""API 响应体模型（统一包一层 code/message，前端好处理）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from app.models.document import ChunkMeta, DocumentMeta

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一响应包装：{"code":0,"message":"ok","data":{...}}"""

    code: int = 0
    message: str = "ok"
    data: T | None = None


class SourceRef(BaseModel):
    """回答引用的原文片段（前端做"查看出处"用）。"""

    chunk_id: str
    document_id: str
    document_title: str = ""
    heading_path: str = ""
    page_no: int = 0
    score: float = 0.0
    content: str = ""          # 截断后的片段文本
    retrieval_source: str = ""  # vector | bm25 | fusion | rerank


class QueryResponse(BaseModel):
    """POST /api/query 响应体。"""

    answer: str
    session_id: str
    sources: list[SourceRef] = Field(default_factory=list)
    transformed_queries: list[str] = Field(default_factory=list)
    cached: bool = False
    latency_ms: int = 0
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    """POST /api/ingest/upload 响应体。"""

    document: DocumentMeta
    chunk_count: int = 0
    elapsed_sec: float = 0.0
    skipped: bool = Field(False, description="文件哈希已存在且已索引时为 True")


class DocumentListResponse(BaseModel):
    """GET /api/ingest/documents 响应体。"""

    total: int
    items: list[DocumentMeta]


class ChunkListResponse(BaseModel):
    """GET /api/ingest/document/{id}/chunks 响应体。"""

    total: int
    items: list[ChunkMeta]


class ReindexResponse(BaseModel):
    """POST /api/ingest/reindex 响应体。"""

    documents: int = 0
    chunks: int = 0
    elapsed_sec: float = 0.0
    failed: list[str] = Field(default_factory=list)


class EvalResponse(BaseModel):
    """评估接口响应体。"""

    run_name: str
    eval_type: str
    metrics: dict[str, float] = Field(default_factory=dict)
    dataset_size: int = 0
    duration_sec: float = 0.0
    detail: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """GET /health 响应体。"""

    status: str = "ok"
    app: str = ""
    environment: str = ""
    version: str = "0.2.0"
    database: bool = False
    redis: bool = False
    cache_backend: str = "memory"
    llm_configured: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
