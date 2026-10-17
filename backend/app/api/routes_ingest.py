"""导入相关接口：上传、删除、重建索引、资料列表。

接口一览：
    POST   /api/ingest/upload                上传文件（PDF/Word/HTML/图片）
    POST   /api/ingest/url                   抓取网页法规
    POST   /api/ingest/text                  粘贴文本建库
    GET    /api/ingest/documents             资料列表
    GET    /api/ingest/document/{id}/chunks  某文档的切块（调试/看切块质量）
    DELETE /api/ingest/document/{id}         删除文档及其向量
    POST   /api/ingest/reindex               重建索引
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import Settings, settings
from app.db.models import Chunk, Document
from app.db.session import get_db
from app.db.vector_store import delete_document_nodes, drop_vector_table
from app.models.document import DocumentMeta
from app.models.request import IngestTextRequest, IngestUrlRequest, ReindexRequest
from app.models.response import (
    ApiResponse,
    ChunkListResponse,
    DocumentListResponse,
    ReindexResponse,
    UploadResponse,
)
from app.utils.cache import clear_cache
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _hype_llm():
    """HyPE 需要 LLM 才能生成假想问题；不传的话 indexer 会静默跳过（曾漏掉这个参数）。

    只在 ENABLE_HYPE=true 时加载，避免无谓地把 LLM 客户端也拉起来。
    """
    if not settings.enable_hype:
        logger.info("HyPE 未启用（ENABLE_HYPE=false），跳过假想问题生成")
        return None
    from app.dependencies import llm_dep

    return llm_dep()


def _save_upload(file: UploadFile) -> Path:
    """把上传文件落到 UPLOAD_DIR，返回落盘路径。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"不支持的文件类型 {suffix}，支持：{sorted(ALLOWED_SUFFIXES)}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{int(time.time())}_{Path(file.filename or 'upload').name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = dest.stat().st_size / 1024 / 1024
    if size_mb > settings.max_upload_mb:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"文件 {size_mb:.1f}MB 超过上限 {settings.max_upload_mb}MB")
    logger.info("已保存上传文件 %s（%.1fMB）", dest.name, size_mb)
    return dest


@router.post("/upload", response_model=ApiResponse[UploadResponse], summary="上传并索引文档")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF / DOCX / HTML / 图片"),
    title: str | None = Form(None),
    doc_category: str = Form("general", description="law | plan | training | general"),
    sync: bool = Form(True, description="true=同步等待完成（小文件）；false=后台索引（大文件）"),
    db: Session = Depends(get_db),
    cfg: Settings = Depends(lambda: settings),
) -> ApiResponse[UploadResponse]:
    """上传并建索引。

    注意：扫描件 OCR + HyPE 会让单文件耗时到分钟级，生产建议 `sync=false`
    走 BackgroundTasks（或换 Celery），前端轮询文档状态。
    """
    from app.core.ingestion.indexer import index_file

    dest = _save_upload(file)

    if not sync:
        # TODO(第二批): 改为投递到任务队列，并把 document 状态机写完整
        def _bg() -> None:
            from app.db.session import SessionLocal

            with SessionLocal() as bg_db:
                try:
                    index_file(bg_db, dest, title=title, doc_category=doc_category, llm=_hype_llm())
                except Exception:  # noqa: BLE001
                    logger.exception("后台索引失败 %s", dest.name)

        background_tasks.add_task(_bg)
        return ApiResponse(
            data=UploadResponse(document=DocumentMeta(id="", title=title or dest.stem, status="pending")),
            message="已提交后台索引，请轮询 /api/ingest/documents 查看状态",
        )

    start = time.perf_counter()
    try:
        doc, chunk_count, skipped = index_file(
            db, dest, title=title, doc_category=doc_category, llm=_hype_llm()
        )
    except Exception as exc:
        raise HTTPException(500, f"索引失败: {exc}") from exc

    # 知识库变了，语义缓存必须失效，否则会返回旧答案
    clear_cache()
    return ApiResponse(
        data=UploadResponse(
            document=DocumentMeta.model_validate(doc),
            chunk_count=chunk_count,
            elapsed_sec=round(time.perf_counter() - start, 2),
            skipped=skipped,
        ),
        message="文件已存在且已索引，跳过" if skipped else "索引完成",
    )


@router.post("/url", response_model=ApiResponse[UploadResponse], summary="抓取网页并索引")
def ingest_url(payload: IngestUrlRequest, db: Session = Depends(get_db)) -> ApiResponse[UploadResponse]:
    from app.core.ingestion.indexer import index_url

    try:
        doc, count = index_url(db, payload.url, title=payload.title, doc_category=payload.doc_category)
    except Exception as exc:
        raise HTTPException(500, f"抓取/索引失败: {exc}") from exc
    clear_cache()
    return ApiResponse(data=UploadResponse(document=DocumentMeta.model_validate(doc), chunk_count=count))


@router.post("/text", response_model=ApiResponse[UploadResponse], summary="粘贴文本建库")
def ingest_text(payload: IngestTextRequest, db: Session = Depends(get_db)) -> ApiResponse[UploadResponse]:
    from app.core.ingestion.indexer import index_file

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = upload_dir / f"text_{int(time.time())}.txt"
    tmp.write_text(payload.text, encoding="utf-8")
    doc, count, _ = index_file(
        db, tmp, title=payload.title, doc_category=payload.doc_category, llm=_hype_llm()
    )
    clear_cache()
    return ApiResponse(data=UploadResponse(document=DocumentMeta.model_validate(doc), chunk_count=count))


@router.get("/documents", response_model=ApiResponse[DocumentListResponse], summary="资料列表")
def list_documents(
    db: Session = Depends(get_db),
    doc_category: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ApiResponse[DocumentListResponse]:
    q = db.query(Document)
    if doc_category:
        q = q.filter(Document.doc_category == doc_category)
    if status:
        q = q.filter(Document.status == status)
    total = q.count()
    rows = q.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    return ApiResponse(
        data=DocumentListResponse(total=total, items=[DocumentMeta.model_validate(r) for r in rows])
    )


@router.get("/document/{doc_id}/chunks", response_model=ApiResponse[ChunkListResponse], summary="查看切块")
def list_chunks(doc_id: str, limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    from app.models.document import ChunkMeta

    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(limit)
        .all()
    )
    return ApiResponse(data=ChunkListResponse(total=len(rows), items=[ChunkMeta.model_validate(r) for r in rows]))


@router.delete("/document/{doc_id}", response_model=ApiResponse[dict], summary="删除文档")
def delete_document(doc_id: str, db: Session = Depends(get_db)) -> ApiResponse[dict]:
    """删除文档：业务表（级联删 chunks）+ 向量库 + 缓存。"""
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, f"文档不存在: {doc_id}")

    deleted_vectors = 0
    try:
        deleted_vectors = delete_document_nodes(doc_id)
    except Exception as exc:
        logger.warning("删除向量失败（继续删业务数据）: %s", exc)

    db.delete(doc)          # chunks 通过 cascade 一起删
    db.commit()
    clear_cache()

    # TODO(第二批): 同步删除 OCR 缓存与上传原文件（注意别误删其他文档引用的同一文件）
    return ApiResponse(data={"document_id": doc_id, "deleted_vectors": deleted_vectors}, message="已删除")


@router.post("/reindex", response_model=ApiResponse[ReindexResponse], summary="重建索引")
def reindex(payload: ReindexRequest, db: Session = Depends(get_db)) -> ApiResponse[ReindexResponse]:
    """重建索引：重新切块 + 重新向量化（改完 chunk 参数/换 embedding 模型后必做）。"""
    from app.core.ingestion.indexer import index_file

    start = time.perf_counter()
    if payload.drop_existing:
        # 换 embedding 维度时必须先 drop，否则插入报维度错误
        drop_vector_table()

    q = db.query(Document)
    if payload.document_ids:
        q = q.filter(Document.id.in_(payload.document_ids))
    docs = q.all()

    failed: list[str] = []
    chunk_total = 0
    hype_llm = _hype_llm()
    for doc in docs:
        if not doc.source_path or not Path(doc.source_path).exists():
            failed.append(f"{doc.id}: 源文件不存在 {doc.source_path}")
            continue
        try:
            _, count, _ = index_file(
                db, doc.source_path, title=doc.title, doc_category=doc.doc_category,
                force=True, llm=hype_llm,
            )
            chunk_total += count
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{doc.id}: {exc}")

    clear_cache()
    return ApiResponse(
        data=ReindexResponse(
            documents=len(docs), chunks=chunk_total,
            elapsed_sec=round(time.perf_counter() - start, 2), failed=failed,
        ),
        message=f"重建完成，失败 {len(failed)} 个",
    )
