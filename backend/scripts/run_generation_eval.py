"""阶段 3：生成层评估（RAGAS）。

分两步，**都可断点续跑**，中断后重跑不会重做已完成的部分：

  步骤 A 答题：对评估集每条问题跑「检索 → 生成」，产出 answer / contexts，
              逐条追加写入 data/eval/gen_filled.jsonl（这一步是确定性成本，缓存后可复用）
  步骤 B 裁判：把 A 的结果交给 RAGAS，算 context_precision / context_recall / faithfulness

为什么不用 /api/eval/generation 接口：140 条要跑 20~40 分钟，单次 HTTP 请求中途失败就全丢；
这里分步落盘，且答题结果可复用于后续消融实验。

用法：
    python scripts/run_generation_eval.py                    # A + B 全跑
    python scripts/run_generation_eval.py --stage fill       # 只答题
    python scripts/run_generation_eval.py --stage ragas      # 只裁判（读已有答题）
    python scripts/run_generation_eval.py --limit 40         # 只跑前 40 条（先小样本试水）
    python scripts/run_generation_eval.py --metrics faithfulness,context_precision
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GOLDEN = Path("data/eval/retrieval_golden.jsonl")
FILLED = Path("data/eval/runs/gen_filled.jsonl")
RESULT = Path("data/eval/runs/eval_gen_result.json")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def stage_fill(items: list[dict], limit: int | None) -> list[dict]:
    """步骤 A：跑检索 + 生成，逐条落盘。"""
    from app.core.generation.llm import chat_direct
    from app.core.generation.prompt import build_qa_messages
    from app.core.retrieval.hybrid import build_hybrid_retriever
    from app.config import settings
    from app.db.vector_store import build_vector_index

    done = {row["question"]: row for row in load_jsonl(FILLED)}
    todo = [it for it in items if it["question"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"答题：已完成 {len(done)} 条，本次待做 {len(todo)} 条", flush=True)

    if not todo:
        return list(done.values())

    index = build_vector_index()
    retriever = build_hybrid_retriever(index)
    keep = settings.rerank_top_k

    start = time.perf_counter()
    with FILLED.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(todo, 1):
            question = item["question"]
            t = time.perf_counter()
            try:
                nodes = retriever.retrieve(question)
                contexts = [getattr(n, "node", n).get_content() for n in nodes[:keep]]
                answer = chat_direct(build_qa_messages(question, nodes))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(todo)}] ✗ {question[:36]} → {type(exc).__name__}: {exc}", flush=True)
                continue

            row = {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.get("ground_truth", ""),
                "gold_ids": item.get("ground_truth_ids", []),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[question] = row
            if i % 5 == 0 or i == len(todo):
                avg = (time.perf_counter() - start) / i
                eta = avg * (len(todo) - i) / 60
                print(f"  [{i}/{len(todo)}] {question[:30]}… {time.perf_counter() - t:.1f}s  "
                      f"累计 {avg:.1f}s/条，预计还需 {eta:.1f} 分钟", flush=True)

    return list(done.values())


def _free_gpu() -> None:
    """答题阶段结束后释放检索器与模型占用的显存。

    答题阶段会加载 bge-m3（2.2GB）+ reranker（2.3GB），而 RAGAS 裁判还要再加载一份
    embedding（另一个实例）。4090/4060Ti 这类 8GB 卡上不释放会 OOM。
    """
    try:
        from app.core.retrieval.hybrid import _RETRIEVER_CACHE

        _RETRIEVER_CACHE.clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.dependencies import reset_model_caches

        reset_model_caches()
    except Exception:  # noqa: BLE001
        pass
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"显存：已释放，剩余 {free / 1024**3:.1f}GB / {total / 1024**3:.1f}GB", flush=True)
    except Exception:  # noqa: BLE001
        pass


def stage_ragas(rows: list[dict], metrics: list[str], run_name: str, batch: int = 40) -> dict:
    """步骤 B：RAGAS 裁判（分批跑，每批落盘，中断不丢已完成的部分）。"""
    from app.core.evaluation.ragas_eval import (
        build_ragas_dataset_from_pipeline,
        evaluate_generation,
    )

    _free_gpu()
    usable = [r for r in rows if r.get("answer") and r.get("contexts") and r.get("ground_truth")]
    print(f"裁判：可用样本 {len(usable)}/{len(rows)} 条，指标 {metrics}，每批 {batch} 条", flush=True)
    if not usable:
        raise SystemExit("没有可用样本（answer/contexts/ground_truth 缺一不可）")

    batches = [usable[i : i + batch] for i in range(0, len(usable), batch)]
    total = len(usable)
    weighted: dict[str, float] = {m: 0.0 for m in metrics}
    done_n = 0
    start = time.perf_counter()

    for bi, chunk in enumerate(batches, 1):
        print(f"\n--- 第 {bi}/{len(batches)} 批（{len(chunk)} 条）---", flush=True)
        payload = build_ragas_dataset_from_pipeline(chunk)
        part = evaluate_generation(payload, metrics=metrics, run_name=f"{run_name}-b{bi}")
        for m, v in part["metrics"].items():
            weighted[m] = weighted.get(m, 0.0) + float(v) * len(chunk)
        done_n += len(chunk)
        elapsed = time.perf_counter() - start
        eta = elapsed / done_n * (total - done_n) / 60
        print(f"  本批 {part['metrics']}｜累计 {done_n}/{total}，预计还需 {eta:.1f} 分钟", flush=True)
        # 每批落盘：中断后至少保留已完成批次的加权结果
        RESULT.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "eval_type": "generation",
                    "metrics": {k: round(v / max(done_n, 1), 4) for k, v in weighted.items()},
                    "dataset_size": done_n,
                    "duration_sec": round(elapsed, 2),
                    "detail": [],
                    "partial": done_n < total,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    result = {
        "run_name": run_name,
        "eval_type": "generation",
        "metrics": {k: round(v / max(done_n, 1), 4) for k, v in weighted.items()},
        "dataset_size": done_n,
        "duration_sec": round(time.perf_counter() - start, 2),
        "detail": [],
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRAGAS 指标：{result['metrics']}（{result['duration_sec']}s，{done_n} 条）", flush=True)
    print(f"结果已写入 {RESULT}", flush=True)

    # 同步写进 eval_records，便于与检索基线一起对比
    try:
        from app.api.routes_eval import _save_record
        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            _save_record(db, result)
            print("已写入 eval_records", flush=True)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        print(f"（写 eval_records 失败，不影响结果文件）：{type(exc).__name__}: {exc}", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fill", "ragas", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--metrics", default="context_precision,context_recall,faithfulness")
    ap.add_argument("--run-name", default="ragas-baseline-v1")
    ap.add_argument("--batch", type=int, default=40, help="RAGAS 分批大小（每批落盘一次）")
    args = ap.parse_args()

    items = load_jsonl(GOLDEN)
    if not items:
        raise SystemExit(f"评估集为空：{GOLDEN}")
    print(f"评估集 {len(items)} 条（{GOLDEN}）", flush=True)

    rows = load_jsonl(FILLED)
    if args.stage in {"fill", "all"}:
        rows = stage_fill(items, args.limit)
    if args.stage in {"ragas", "all"}:
        stage_ragas(
            rows,
            [m.strip() for m in args.metrics.split(",") if m.strip()],
            args.run_name,
            batch=args.batch,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
