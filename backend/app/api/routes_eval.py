"""评估接口：检索指标（第一批基线）与 RAGAS 生成指标（第二批）。

接口一览：
    POST /api/eval/retrieval   跑 MRR / HitRate / Recall / nDCG，写 eval_records
    POST /api/eval/generation  跑 RAGAS（context_precision/context_recall/faithfulness）
    GET  /api/eval/records     查看历史评估记录（对比调优效果）
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import EvalRecord
from app.db.session import get_db
from app.models.request import GenerationEvalRequest, RetrievalEvalRequest
from app.models.response import ApiResponse, EvalResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])


def _load_dataset(path: str | None, inline: list[dict]) -> list[dict]:
    """评估集来源：优先内联，其次 JSONL 文件。"""
    if inline:
        return inline
    if not path:
        raise HTTPException(400, "必须提供 dataset 或 dataset_path")
    from app.core.evaluation.retriever_eval import load_golden_set

    try:
        return load_golden_set(path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/retrieval", response_model=ApiResponse[EvalResponse], summary="检索评估（MRR / HitRate）")
def eval_retrieval(payload: RetrievalEvalRequest, db: Session = Depends(get_db)):
    """跑检索评估。评估集格式见 app/core/evaluation/retriever_eval.py 顶部注释。"""
    from app.core.evaluation.retriever_eval import evaluate_retrieval
    from app.core.retrieval.hybrid import build_hybrid_retriever
    from app.db.vector_store import build_vector_index

    dataset = _load_dataset(payload.dataset_path, payload.dataset)
    if not dataset:
        raise HTTPException(400, "评估集为空")

    # 冷启动：BM25 语料需要从向量库加载，第一次会慢几秒
    index = build_vector_index()
    retriever = build_hybrid_retriever(index, use_rerank=payload.use_rerank, top_k=payload.top_k)

    result = evaluate_retrieval(retriever, dataset, top_k=payload.top_k, run_name=payload.run_name)
    _save_record(db, result)
    return ApiResponse(data=EvalResponse(**result))


@router.post("/generation", response_model=ApiResponse[EvalResponse], summary="RAGAS 生成评估")
def eval_generation(payload: GenerationEvalRequest, db: Session = Depends(get_db)):
    """RAGAS 评估。dataset 里若没有 answer/contexts，本接口会先跑一遍问答链路补齐。"""
    from app.core.evaluation.ragas_eval import evaluate_generation
    from app.core.generation.llm import chat_direct
    from app.core.generation.prompt import build_qa_messages
    from app.core.retrieval.hybrid import build_hybrid_retriever
    from app.db.vector_store import build_vector_index

    dataset = _load_dataset(payload.dataset_path, payload.dataset)
    if not dataset:
        raise HTTPException(400, "评估集为空")

    # 补齐 answer / contexts：RAGAS 需要这两项
    need_pipeline = any(not item.get("answer") or not item.get("contexts") for item in dataset)
    if need_pipeline:
        index = build_vector_index()
        retriever = build_hybrid_retriever(index)
        for item in dataset:
            if item.get("answer") and item.get("contexts"):
                continue
            nodes = retriever.retrieve(item["question"])
            item["contexts"] = [getattr(n, "node", n).get_content() for n in nodes[: settings.rerank_top_k]]
            if not item.get("answer"):
                item["answer"] = chat_direct(build_qa_messages(item["question"], nodes))

    try:
        result = evaluate_generation(dataset, metrics=payload.metrics, run_name=payload.run_name)
    except ImportError as exc:
        raise HTTPException(
            500,
            f"RAGAS 依赖缺失（需额外 pip install langchain-openai datasets）：{exc}",
        ) from exc
    _save_record(db, result)
    return ApiResponse(data=EvalResponse(**result))


@router.get("/records", response_model=ApiResponse[list[dict]], summary="历史评估记录")
def list_records(
    eval_type: str | None = None,
    run_name: str | None = None,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    q = db.query(EvalRecord)
    if eval_type:
        q = q.filter(EvalRecord.eval_type == eval_type)
    if run_name:
        q = q.filter(EvalRecord.run_name == run_name)
    rows = q.order_by(EvalRecord.created_at.desc()).limit(limit).all()
    return ApiResponse(
        data=[
            {
                "id": r.id,
                "run_name": r.run_name,
                "eval_type": r.eval_type,
                "metrics": r.metrics,
                "dataset_size": r.dataset_size,
                "duration_sec": r.duration_sec,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    )


def _save_record(db: Session, result: dict) -> None:
    """评估结果落库，便于跨时间对比。"""
    record = EvalRecord(
        run_name=result.get("run_name", "default"),
        eval_type=result.get("eval_type", "retrieval"),
        metrics=result.get("metrics", {}),
        dataset_size=result.get("dataset_size", 0),
        duration_sec=result.get("duration_sec", 0.0),
        detail=result.get("detail", []),
        config_snapshot={
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model if settings.rerank_provider != "none" else None,
            "chunk_size": settings.chunk_size,
            "retrieve_top_k": settings.retrieve_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "enable_hype": settings.enable_hype,
            "enable_hyde": settings.enable_hyde,
        },
    )
    db.add(record)
    db.commit()


# ---------------------------------------------------------------- 便捷脚本入口
QUESTION_TYPES = ("事实型", "归纳型", "易混淆型")


def _stratified_sample(db: Session, sample_size: int) -> tuple[list[Any], dict[str, int]]:
    """按文档规模**分层抽样**切块。

    为什么不能直接 `db.query(Chunk).limit(n)`：那是按插入顺序取前 n 个 ——
    三份资料共 540 块时，取 150 只会拿到最早入库的导则与消防法，
    **GB 51283（359 块、出题价值最高）一块都拿不到**。

    做法：按各文档切块数占比分配名额（每份至少 3 条），
    在每份文档内部按等间隔取样，让题目分散到不同章节而不是挤在开头。
    """
    from app.db.models import Chunk as ChunkModel
    from app.db.models import Document as DocumentModel

    docs = (
        db.query(DocumentModel)
        .filter(DocumentModel.status == "indexed")
        .order_by(DocumentModel.chunk_count.desc())
        .all()
    )
    if not docs:
        return [], {}

    total = sum(d.chunk_count or 0 for d in docs) or 1
    picked: list[Any] = []
    plan: dict[str, int] = {}

    for doc in docs:
        quota = max(3, round(sample_size * (doc.chunk_count or 0) / total))
        rows = (
            db.query(ChunkModel)
            .filter(ChunkModel.document_id == doc.id)
            .order_by(ChunkModel.chunk_index)
            .all()
        )
        if not rows:
            continue
        quota = min(quota, len(rows))
        step = len(rows) / quota
        chosen = [rows[int(i * step)] for i in range(quota)]
        picked.extend(chosen)
        plan[f"{doc.title}（{doc.chunk_count}块）"] = len(chosen)

    return picked, plan


@router.post("/build-golden-set", response_model=ApiResponse[dict], summary="用 LLM 自动生成评估集草稿")
def build_golden_set(sample_size: int = 30, db: Session = Depends(get_db)):
    """从库里分层抽样切块，让 DeepSeek 出题，生成待人工校对的评估集草稿。

    ⚠️ 自动生成的 ground_truth 必须人工过一遍，否则评估基线不可信。
    输出带的 type 字段（事实型/归纳型/易混淆型）用于人工按策略配比挑选。
    """
    import json

    from app.core.generation.llm import chat_direct

    chunks, plan = _stratified_sample(db, sample_size)
    if not chunks:
        raise HTTPException(400, "知识库为空，先导入资料")

    items: list[dict] = []
    failed = 0
    for row in chunks:
        prompt = (
            "根据下面的消防资料片段，出一道工厂员工会问的问题，并给出标准答案。\n"
            "要求：type 只能是【事实型】（问具体数值/条款）、【归纳型】（需综合多条内容）、"
            "【易混淆型】（针对容易混淆的概念）三者之一。\n"
            '只输出 JSON：{"question":"...","ground_truth":"...","type":"事实型"}\n\n'
            f"片段：\n{row.content[:1500]}"
        )
        try:
            raw = chat_direct([{"role": "user", "content": prompt}]).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            obj = json.loads(raw)
            qtype = str(obj.get("type", "")).strip()
            items.append(
                {
                    "question": obj["question"],
                    "ground_truth": obj["ground_truth"],
                    "ground_truth_ids": [row.id],
                    "type": qtype if qtype in QUESTION_TYPES else "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning("生成题目失败: %s", exc)

    out_path = Path("./data/eval/retrieval_golden_draft.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    return ApiResponse(
        data={
            "count": len(items),
            "failed": failed,
            "plan": plan,
            "type_dist": {t: sum(1 for i in items if i["type"] == t) for t in QUESTION_TYPES},
            "path": str(out_path),
        },
        message="草稿已生成，请人工校对",
    )
