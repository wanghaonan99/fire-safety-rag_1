"""文档相关的 Pydantic 领域模型（与 ORM 解耦，API 层只暴露这些）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DocCategory = Literal["law", "plan", "training", "general"]
DocStatus = Literal["pending", "parsing", "ocr", "chunking", "embedding", "indexed", "failed"]
SourceType = Literal["pdf", "docx", "doc", "html", "htm", "txt", "md", "image"]


class DocumentMeta(BaseModel):
    """一篇文档的元信息。"""

    id: str
    title: str
    source_type: SourceType = "pdf"
    source_path: str = ""
    file_hash: str = ""
    doc_category: DocCategory = "general"
    status: DocStatus = "pending"
    is_scanned: bool = False
    page_count: int = 0
    chunk_count: int = 0
    error_message: str = ""
    # ⚠️ 命名冲突：ORM 类继承自 SQLAlchemy 的 Base，而 Base 上已经有 `metadata`
    #    （那是 MetaData 对象）。所以 ORM 侧我们的列叫 doc_metadata，
    #    这里必须用 validation_alias 指过去，否则 model_validate(orm) 会读到 MetaData
    #    并报 "Input should be a valid dictionary [input_value=MetaData()]"。
    #    serialization_alias 保证对外的 JSON 字段名仍是 metadata。
    metadata: dict[str, Any] = Field(
        default_factory=dict, validation_alias="doc_metadata", serialization_alias="metadata"
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ChunkMeta(BaseModel):
    """一个切块及其溯源信息。"""

    id: str
    document_id: str
    chunk_index: int = 0
    content: str
    heading_path: str = ""
    token_count: int = 0
    page_no: int = 0
    hypothetical_questions: list[str] = Field(default_factory=list)
    # 同 DocumentMeta.metadata 的原因
    metadata: dict[str, Any] = Field(
        default_factory=dict, validation_alias="chunk_metadata", serialization_alias="metadata"
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ParsedDocument(BaseModel):
    """解析器的统一中间产物（所有 loader 都返回它，便于替换解析实现）。"""

    title: str
    text: str
    source_type: SourceType = "pdf"
    page_count: int = 0
    is_scanned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
