"""检索评估：自建 MRR / Hit Rate / Recall@K / nDCG。

评估集格式（data/eval/retrieval_golden.jsonl，每行一条）：
    {"question": "甲类厂房防火间距是多少？", "ground_truth_ids": ["chunk-uuid-1"], "ground_truth": "..."}

两个核心指标：
- **Hit Rate@K**：前 K 条里是否命中任一正确片段（衡量"能不能找到"，看召回）；
- **MRR@K**：第一个正确片段的排名倒数的均值（衡量"排得好不好"，看排序质量）。
调优 Hybrid/Rerank 时主要看 MRR 的涨跌。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


def load_golden_set(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 评估集。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评估集不存在: {p}")
    items: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        items.append(json.loads(line))
    logger.info("加载评估集 %s 条：%s", len(items), p)
    return items


def _node_ids(nodes: list[Any]) -> list[str]:
    """从检索结果里取 chunk_id（优先 metadata.chunk_id，回退 node_id）。"""
    ids: list[str] = []
    for n in nodes:
        node = getattr(n, "node", n)
        meta = node.metadata or {}
        ids.append(str(meta.get("chunk_id") or node.node_id))
    return ids


def _relaxed_hit(retrieved_id: str, gold_ids: list[str]) -> bool:
    """命中判定：完全相等，或 gold 是 retrieved 的前缀片段之外仍算命中。

    实际评估里 ground_truth 常标注到"页面/段落"级，而 chunk 粒度更细，
    这里做一层宽松匹配（gold 出现在检索结果里或反之）。
    """
    for g in gold_ids:
        if retrieved_id == g or g in retrieved_id or retrieved_id in g:
            return True
    return False


def evaluate_retrieval(
    retriever: Any,
    dataset: list[dict[str, Any]],
    top_k: int = 10,
    run_name: str = "retrieval-baseline",
) -> dict[str, Any]:
    """跑一遍评估集，返回指标与逐条明细。"""
    start = time.perf_counter()
    hits = 0
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    detail: list[dict[str, Any]] = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        gold_ids = [str(x) for x in item.get("ground_truth_ids", [])]
        try:
            nodes = retriever.retrieve(question)
        except Exception as exc:
            logger.warning("检索失败 [%s] %s: %s", i, question, exc)
            nodes = []
        retrieved = _node_ids(nodes)[:top_k]

        # Hit Rate
        first_rank = 0
        for rank, rid in enumerate(retrieved, 1):
            if _relaxed_hit(rid, gold_ids):
                first_rank = rank
                break
        hit = first_rank > 0
        hits += int(hit)

        # MRR
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

        # Recall@K（多标注场景）
        found = sum(1 for g in gold_ids if any(_relaxed_hit(r, [g]) for r in retrieved))
        recalls.append(found / len(gold_ids) if gold_ids else 0.0)

        # nDCG@K（相关性用 0/1）
        dcg = sum(1 / math.log2(r + 1) for r, rid in enumerate(retrieved, 1) if _relaxed_hit(rid, gold_ids))
        idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold_ids), top_k) + 1)) if gold_ids else 1.0
        ndcgs.append(dcg / idcg if idcg else 0.0)

        detail.append(
            {
                "question": question,
                "hit": hit,
                "first_rank": first_rank,
                "gold_ids": gold_ids,
                "retrieved_ids": retrieved,
            }
        )
        if i % 10 == 0:
            logger.info("评估进度 %s/%s，当前 HitRate=%.3f", i, len(dataset), hits / i)

    n = max(len(dataset), 1)
    metrics = {
        f"hit_rate@{top_k}": round(hits / n, 4),
        f"mrr@{top_k}": round(sum(reciprocal_ranks) / n, 4),
        f"recall@{top_k}": round(sum(recalls) / n, 4),
        f"ndcg@{top_k}": round(sum(ndcgs) / n, 4),
    }
    duration = time.perf_counter() - start
    logger.info("检索评估完成：%s（%.1fs）", metrics, duration)

    return {
        "run_name": run_name,
        "eval_type": "retrieval",
        "metrics": metrics,
        "dataset_size": len(dataset),
        "duration_sec": round(duration, 2),
        "detail": detail,
    }


def compare_runs(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    """对比两次评估的指标变化，用于判断"这次调优到底有没有用"。"""
    return {k: round(candidate.get(k, 0.0) - baseline.get(k, 0.0), 4) for k in baseline}
