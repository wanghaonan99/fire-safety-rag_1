"""关系型数据模型（文档 / 块 / 会话 / 消息 / 评估记录）。

与 LlamaIndex 的分工：
- 本文件这几张表是**业务元数据**，用于管理、溯源、会话记忆、评估留痕；
- 向量本身由 LlamaIndex 的 PGVectorStore 写入它自建的 `data_<table_name>` 表，
  主键用 `chunk_id` 与这里的 `chunks.id` 对应（metadata 里携带）。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def _uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Document(Base, TimestampMixin):
    """一篇被导入的原始资料（法规 / 预案 / 培训课件 / 扫描件）。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), default="pdf")     # pdf/docx/html/image
    source_path: Mapped[str] = mapped_column(Text, default="")
    file_hash: Mapped[str] = mapped_column(String(64), default="", index=True)  # 幂等去重用
    doc_category: Mapped[str] = mapped_column(String(64), default="general")    # 法规/预案/培训
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    # pending | parsing | ocr | chunking | embedding | indexed | failed
    is_scanned: Mapped[bool] = mapped_column(Boolean, default=False)        # 是否走了 OCR 降级
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    error_message: Mapped[str] = mapped_column(Text, default="")

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_documents_hash_status", "file_hash", "status"),)


class Chunk(Base, TimestampMixin):
    """文档切块。同时保存 HyPE 生成的假想问题，供检索与评估使用。"""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[str] = mapped_column(Text, default="")             # 标题链，如 "3.2 疏散通道 > 3.2.1"
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    # HyPE：建库时让 LLM 生成 3 个"这段文字能回答的问题"，写进 metadata 一起被索引
    hypothetical_questions: Mapped[list] = mapped_column(JSONB, default=list)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    # 保留列：走"自建检索"而非 LlamaIndex 时使用；默认流程不写它
    embedding: Mapped[list | None] = mapped_column(Vector(settings.embedding_dim), nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChatSession(Base, TimestampMixin):
    """一次会话（浏览器端用一个 session_id 串起来）。"""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), default="anonymous", index=True)
    title: Mapped[str] = mapped_column(String(255), default="新会话")
    summary: Mapped[str] = mapped_column(Text, default="")                  # 超出窗口后的滚动摘要
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    session_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    """会话中的一轮消息（user / assistant），用于上下文记忆。"""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)           # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list] = mapped_column(JSONB, default=list)              # 本条回答引用的 chunk_id
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_chat_messages_session_created", "session_id", "created_at"),)


class EvalRecord(Base, TimestampMixin):
    """评估留痕：检索指标（MRR/HitRate）与生成指标（RAGAS）各存一行。"""

    __tablename__ = "eval_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_name: Mapped[str] = mapped_column(String(128), default="default", index=True)
    eval_type: Mapped[str] = mapped_column(String(32), nullable=False)      # retrieval | generation
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)              # {"mrr":0.82,...}
    dataset_size: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)      # 当时的检索/模型参数
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[list] = mapped_column(JSONB, default=list)               # 逐条样本结果，便于排查
