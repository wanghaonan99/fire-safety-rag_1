"""建索引流水线：解析 -> 切块 -> HyPE -> 向量化入库 -> 落业务表。

这是 ingest 路由背后的"大函数"，单独成模块便于：
1. CLI 批量导入（scripts 里直接调 `index_file`）；
2. 单元测试替换 embed_model / llm 为假实现；
3. 未来换成异步任务队列（Celery / Render Cron）时只改调用方。

注：本模块是**对用户给定目录树的新增**（原树未列出），
因为 ingestion 目录给的是各步工具函数，需要一个编排者把它们串起来。
"""

from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.core.ingestion.chunker import chunk_document
from app.core.ingestion.embedder import build_embed_model
from app.core.ingestion.hyphe import attach_hype
from app.core.ingestion.loader import file_hash, load_file
from app.db.models import Chunk, Document
from app.utils.logging import get_logger

logger = get_logger(__name__)


def index_file(
    db: Session,
    file_path: str | Path,
    title: str | None = None,
    doc_category: str = "general",
    use_ocr_fallback: bool = True,
    embed_model=None,
    llm=None,
    force: bool = False,
) -> tuple[Document, int, bool]:
    """把一个文件导入知识库。

    返回 (Document 记录, chunk 数, 是否因已存在而跳过)
    """
    from app.db.vector_store import build_vector_index, delete_document_nodes

    path = Path(file_path)
    fhash = file_hash(path)

    # ---------- 1. 幂等检查 / 强制重建 ----------
    existing = (
        db.query(Document)
        .filter(Document.file_hash == fhash, Document.status == "indexed")
        .first()
    )
    if existing and not force:
        logger.info("文件已索引，跳过：%s", path.name)
        return existing, existing.chunk_count, True

    if existing and force:
        # force 的语义是"重建这份文档"，而不是再插一份重复的：
        # 必须连同旧向量一起清掉，否则库里会留下两份内容相同的文档。
        logger.info("force=True，先清除旧版本：%s（%s 块）", existing.title, existing.chunk_count)
        try:
            delete_document_nodes(existing.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("清除旧向量失败（继续）: %s", exc)
        db.delete(existing)          # chunks 由 cascade 一并删除
        db.commit()

    doc = Document(
        title=title or path.stem,
        source_type=path.suffix.lstrip(".").lower(),
        source_path=str(path),
        file_hash=fhash,
        doc_category=doc_category,
        status="parsing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    start = time.perf_counter()
    try:
        # ---------- 2. 解析（含扫描件 OCR 降级）----------
        parsed = load_file(path, use_ocr_fallback=use_ocr_fallback)
        doc.title = title or parsed.title or doc.title
        doc.is_scanned = parsed.is_scanned
        doc.page_count = parsed.page_count
        doc.doc_metadata = {**parsed.metadata, "source_type": parsed.source_type}
        doc.status = "chunking"
        db.commit()

        # ---------- 3. 切块（语义切块 + 标题拼接）----------
        if embed_model is not None:
            embed = embed_model
        else:
            # 单例：新建实例等于重新加载 2.2GB 模型
            from app.dependencies import embed_model_dep

            embed = embed_model_dep()
        chunks = chunk_document(parsed, document_id=doc.id, embed_model=embed, doc_category=doc_category)
        if not chunks:
            raise ValueError("切块结果为空，请检查文档内容")

        # ---------- 4. HyPE：为每个 chunk 生成假想问题 ----------
        doc.status = "embedding"
        db.commit()
        chunks = attach_hype(chunks, llm=llm)

        # ---------- 5. 落业务表并拿回稳定 chunk_id ----------
        db_chunks: list[Chunk] = []
        for c in chunks:
            row = Chunk(
                document_id=doc.id,
                chunk_index=c["chunk_index"],
                content=c["content"],
                heading_path=c.get("heading_path", ""),
                token_count=c.get("token_count", 0),
                page_no=c.get("page_no", 0),
                hypothetical_questions=(c.get("metadata") or {}).get("hypothetical_questions", []),
                chunk_metadata=c.get("metadata", {}),
            )
            db.add(row)
            db_chunks.append(row)
        db.flush()   # 拿到自增/生成的 id

        # ---------- 6. 构造 LlamaIndex 节点并写入 pgvector ----------
        nodes = _build_nodes(db_chunks, doc)
        build_vector_index(nodes=nodes, embed_model=embed)

        # ---------- 7. 收尾 ----------
        doc.chunk_count = len(db_chunks)
        doc.status = "indexed"
        doc.error_message = ""
        db.commit()

        # 知识库变了：检索器缓存（含 BM25 语料）与索引单例必须失效，否则新数据查不到
        from app.core.retrieval.hybrid import invalidate_retriever_cache
        from app.db.vector_store import invalidate_index_cache

        invalidate_index_cache()
        invalidate_retriever_cache()

        elapsed = time.perf_counter() - start
        logger.info("导入完成 %s：%s 块，耗时 %.1fs", doc.title, doc.chunk_count, elapsed)
        return doc, doc.chunk_count, False

    except Exception as exc:
        db.rollback()
        doc.status = "failed"
        doc.error_message = str(exc)[:1000]
        db.commit()
        # 失败时清掉可能已写入的部分向量，保持一致性
        try:
            delete_document_nodes(doc.id)
        except Exception:  # noqa: BLE001
            pass
        logger.exception("导入失败 %s: %s", path.name, exc)
        raise


def _build_nodes(db_chunks: list[Chunk], doc: Document) -> list:
    """DB chunk 行 -> LlamaIndex TextNode（含 HyPE 增强索引文本）。

    ⚠️ 关于 metadata 里的 `document_id`：LlamaIndex 的 `node_to_metadata_dict()` 会用
    节点自身的 `ref_doc_id` **覆盖**同名键，并把 None 序列化成字符串 `'None'`
    （实测：传进去 'REAL-UUID' 出来还是 'None'）。所以必须按它的语义设置
    NodeRelationship.SOURCE，而不是自己往 metadata 里塞 document_id——
    否则向量表里的 document_id 永远是 'None'，按它删除/过滤全都失效。
    """
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    nodes = []
    for row in db_chunks:
        meta = {
            **(row.chunk_metadata or {}),
            "chunk_id": row.id,
            "title": doc.title,
            "heading_path": row.heading_path,
            "page_no": row.page_no,
            "doc_category": doc.doc_category,
            "hypothetical_questions": row.hypothetical_questions or [],
        }
        node = TextNode(
            id_=row.id,
            text=row.content,
            metadata=meta,
            # 展示/生成用原文；HyPE 的假想问题放在 metadata 里，**不能**排除出嵌入元数据，
            # 否则不会被向量化（LlamaIndex 嵌入的是 node.get_content(EMBED)，
            # 包含所有未排除的 metadata），HyPE 就白做了。
            excluded_embed_metadata_keys=["title", "doc_category"],
            excluded_llm_metadata_keys=["hypothetical_questions"],
        )
        # 让 LlamaIndex 知道这段来自哪份文档（会写进向量表的 document_id 字段）
        node.relationships = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=str(doc.id))}
        nodes.append(node)
    return nodes


def index_url(db: Session, url: str, title: str | None = None, doc_category: str = "law") -> tuple[Document, int]:
    """抓网页并入库（消防法规多发布在政府网站）。

    TODO(第二批): 用 playwright 抓 JS 渲染页面；限速 + robots.txt 检查。
    """
    import httpx

    from app.core.ingestion.loader import load_html

    resp = httpx.get(url, timeout=30.0, headers={"User-Agent": "fire-safety-rag/0.2"})
    resp.raise_for_status()
    parsed = load_html(html=resp.text, title=title or url)

    tmp_dir = Path(settings.upload_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"{abs(hash(url))}.html"
    tmp_file.write_text(parsed.text, encoding="utf-8")

    doc, count, _ = index_file(db, tmp_file, title=parsed.title, doc_category=doc_category)
    doc.source_type = "html"
    doc.doc_metadata = {**(doc.doc_metadata or {}), "url": url}
    db.commit()
    return doc, count
