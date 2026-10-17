"""分析检索评估结果：总分 + 分资料 + 排名分布 + 跨文档串味。

用法：python scripts/analyze_eval_result.py [结果JSON] [--list N]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0] if args else "data/eval/runs/eval_baseline_result.json")
    show = 12
    if "--list" in sys.argv:
        idx = sys.argv.index("--list")
        show = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 12

    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    metrics = data["metrics"]
    detail = data["detail"]

    print("=" * 84)
    print(f"运行：{data['run_name']}   样本 {data['dataset_size']} 条   耗时 {data['duration_sec']}s")
    print("=" * 84)
    for k, v in metrics.items():
        print(f"  {k:<16} {v:.4f}")

    # 取切块->文档映射，用于分资料统计与"串味"判定
    with engine.connect() as c:
        doc_of = {
            r[0]: r[1]
            for r in c.execute(text("SELECT id::text, document_id::text FROM chunks")).all()
        }
        titles = {r[0]: r[1] for r in c.execute(text("SELECT id::text, title FROM documents")).all()}

    def doc(cid: str) -> str:
        return titles.get(doc_of.get(cid, ""), "?")

    # 分资料统计
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for d in detail:
        gold = d["gold_ids"][0] if d["gold_ids"] else ""
        by_doc[titles.get(doc_of.get(gold, ""), "?")[:30]].append(d)

    print("\n按来源资料：")
    print(f"  {'资料':<34}{'条数':>5}{'HitRate':>9}{'MRR':>8}{'排第1':>7}")
    for name, items in sorted(by_doc.items(), key=lambda x: -len(x[1])):
        n = len(items)
        hits = sum(1 for i in items if i["hit"])
        mrr = sum(1.0 / i["first_rank"] if i["first_rank"] else 0.0 for i in items) / n
        rank1 = sum(1 for i in items if i["first_rank"] == 1)
        print(f"  {name:<34}{n:>5}{hits / n:>9.3f}{mrr:>8.3f}{rank1 / n:>7.1%}")

    # 排名分布
    print("\n命中排名分布（第一条正确结果排在第几）：")
    ranks = Counter(d["first_rank"] if d["hit"] else 0 for d in detail)
    total = len(detail)
    for r in sorted(ranks):
        label = "未命中" if r == 0 else f"第 {r} 名"
        bar = "█" * int(ranks[r] / total * 40)
        print(f"  {label:<8}{ranks[r]:>4} ({ranks[r] / total:>5.1%}) {bar}")

    # 跨文档串味：排第 1 的结果与正确答案不在同一份资料
    cross: list[dict] = []
    for d in detail:
        if not d["retrieved_ids"] or not d["gold_ids"]:
            continue
        gold_doc = doc(d["gold_ids"][0])
        top_doc = doc(d["retrieved_ids"][0])
        if gold_doc != "?" and top_doc != "?" and gold_doc != top_doc:
            cross.append({**d, "gold_doc": gold_doc, "top_doc": top_doc})
    print(f"\n跨文档串味（Top-1 来自别的资料）：{len(cross)}/{total} = {len(cross) / total:.1%}")

    # 需要改进的条目
    weak = [d for d in detail if not d["hit"] or d["first_rank"] > 1]
    weak.sort(key=lambda d: (d["hit"], d["first_rank"]))
    print(f"\n排名不理想（未命中或非第 1 名）的条目：{len(weak)} 条，列前 {min(show, len(weak))} 条")
    for d in weak[:show]:
        tag = "未命中" if not d["hit"] else f"第{d['first_rank']}名"
        gold_doc = doc(d["gold_ids"][0])[:16] if d["gold_ids"] else "?"
        top_doc = doc(d["retrieved_ids"][0])[:16] if d["retrieved_ids"] else "-"
        flag = " ⚠️串味" if cross and any(c["question"] == d["question"] for c in cross) else ""
        print(f"  [{tag}] {d['question'][:52]}{flag}")
        print(f"        正确来源《{gold_doc}》· Top1 来自《{top_doc}》")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
