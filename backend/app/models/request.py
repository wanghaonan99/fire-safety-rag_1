"""API 请求体模型（POST 入参统一在这里定义）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /api/query 请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(None, description="会话 id；不传则自动新建")
    user_id: str = Field("anonymous", description="用户标识，用于会话隔离")
    top_k: int | None = Field(None, ge=1, le=50, description="最终送入 LLM 的片段数")
    doc_category: str | None = Field(None, description="限定资料类别：law/plan/training")
    use_cache: bool = Field(True, description="是否走语义缓存")
    use_query_transform: bool | None = Field(None, description="覆盖默认的查询改写开关")
    stream: bool = Field(False, description="是否流式返回（SSE）")


class IngestUrlRequest(BaseModel):
    """POST /api/ingest/url 请求体（抓网页版消防法规）。"""

    url: str
    title: str | None = None
    doc_category: str = "law"


class IngestTextRequest(BaseModel):
    """POST /api/ingest/text 请求体（粘贴一段文本快速建库）。"""

    title: str
    text: str
    doc_category: str = "general"


class ReindexRequest(BaseModel):
    """POST /api/ingest/reindex 请求体。"""

    document_ids: list[str] = Field(default_factory=list, description="为空表示全量重建")
    drop_existing: bool = Field(False, description="是否先清空向量表再重建")


class RetrievalEvalRequest(BaseModel):
    """POST /api/eval/retrieval 请求体：用带标注的问答集评估检索器。"""

    dataset_path: str | None = Field(None, description="JSONL 路径；与 dataset 二选一")
    dataset: list[dict[str, Any]] = Field(default_factory=list, description="[{question, ground_truth_ids}]")
    top_k: int = Field(10, ge=1, le=50)
    run_name: str = "retrieval-baseline"
    use_rerank: bool = True


class GenerationEvalRequest(BaseModel):
    """POST /api/eval/generation 请求体：RAGAS 评估端到端生成质量。"""

    dataset_path: str | None = None
    dataset: list[dict[str, Any]] = Field(default_factory=list, description="[{question, ground_truth}]")
    metrics: list[Literal["context_precision", "context_recall", "faithfulness", "answer_relevancy"]] = Field(
        default_factory=lambda: ["context_precision", "context_recall", "faithfulness"]
    )
    run_name: str = "ragas-baseline"


class LoginRequest(BaseModel):
    """POST /api/admin/login 请求体（第一批只做最简单的管理口令）。"""

    password: str
