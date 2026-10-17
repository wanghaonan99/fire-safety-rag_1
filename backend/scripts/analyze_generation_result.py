"""分析 RAGAS 结果：总体 + 分资料 + 最差样本清单（用于定位 faithfulness 低的原因）。

用法：
    python scripts/analyze_generation_result.py                     # 用 data/eval_gen_result.json
    python scripts/analyze_generation_result.py <结果> --list 15
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402

THRESHOLDS = {"faithfulness": 0.90, "context_precision": 0.75, "context_recall": 0.75}
GOLDEN = Path("data/eval/retrieval_golden.jsonl")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0] if args else "data/eval/runs/eval_gen_result.json")
    show = 15
    if "--list" in sys.argv:
        i = sys.argv.index("--list")
        show = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 15

    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    detail = data.get("detail") or []

    print("=" * 86)
    print(f"运行：{data.get('run_name')}   样本 {data.get('dataset_size')} 条   "
          f"耗时 {data.get('duration_sec')}s" + ("（分批累计）" if data.get("partial") else ""))
    print("=" * 86)
    for k, v in metrics.items():
        thr = THRESHOLDS.get(k)
        mark = "" if thr is None else ("  ✅ 达标" if v >= thr else f"  ❌ 未达门槛 {thr}")
        print(f"  {k:<20} {v:.4f}{mark}")

    if not detail:
        print("\n（本次结果没有逐条得分：旧版本导出为空，需要重跑一次带 detail 的评估）")
        return 0

    # 问题 -> 来源资料（通过评估集里的 ground_truth_ids 反查）
    golden = {}
    if GOLDEN.exists():
        for line in GOLDEN.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                golden[(row.get("question") or "").strip()] = row.get("ground_truth_ids") or []

    with engine.connect() as c:
        doc_of = {r[0]: r[1] for r in c.execute(text("SELECT id::text, document_id::text FROM chunks")).all()}
        titles = {r[0]: r[1] for r in c.execute(text("SELECT id::text, title FROM documents")).all()}

    def doc_name(question: str) -> str:
        ids = golden.get(question.strip()) or []
        if not ids:
            return "?"
        return titles.get(doc_of.get(str(ids[0]), ""), "?")[:30]

    # 分资料统计 faithfulnes
    by_doc: dict[str, list[float]] = defaultdict(list)
    for d in detail:
        f = d["scores"].get("faithfulness")
        if f is not None:
            by_doc[doc_name(d["question"])].append(f)

    print("\n按来源资料（faithfulness）：")
    for name, vals in sorted(by_doc.items(), key=lambda x: -len(x[1])):
        avg = sum(vals) / len(vals)
        low = sum(1 for v in vals if v < 0.5)
        print(f"  {name:<32} n={len(vals):>3}  均值 {avg:.3f}   低于0.5的 {low} 条")

    # 最差样本
    scored = [d for d in detail if d["scores"].get("faithfulness") is not None]
    scored.sort(key=lambda d: d["scores"]["faithfulness"])
    print(f"\nfaithfulness 最低的 {min(show, len(scored))} 条：")
    for d in scored[:show]:
        f = d["scores"]["faithfulness"]
        print(f"\n  [{f:.2f}] {d['question'][:64]}")
        print(f"        来源《{doc_name(d['question'])}》· 上下文 {d['n_contexts']} 段")
        print(f"        答案：{d['answer'][:190].replace(chr(10), ' ')}")

    # 分数分布
    buckets = {"1.0（完全有据）": 0, "0.8~1.0": 0, "0.5~0.8": 0, "0.0~0.5（严重）": 0}
    for d in scored:
        f = d["scores"]["faithfulness"]
        if f >= 0.999:
            buckets["1.0（完全有据）"] += 1
        elif f >= 0.8:
            buckets["0.8~1.0"] += 1
        elif f >= 0.5:
            buckets["0.5~0.8"] += 1
        else:
            buckets["0.0~0.5（严重）"] += 1
    print("\nfaithfulness 分布：")
    for k, v in buckets.items():
        print(f"  {k:<20} {v:>4} 条（{v / max(len(scored), 1):.1%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
