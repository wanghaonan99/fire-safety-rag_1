"""把勾选清单转换成最终评估集（只取 _keep=true 的条目）。

工作流：
    1) 打开 data/eval/retrieval_golden_preselect.jsonl，逐条把 "_keep" 改成 true/false
    2) 运行本脚本 → 生成 data/eval/retrieval_golden.jsonl（只含 3 个字段，供评估接口使用）

用法：
    cd backend
    python scripts/build_golden_set.py                    # 默认输入输出
    python scripts/build_golden_set.py <输入> <输出>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_IN = "data/eval/retrieval_golden_preselect.jsonl"
DEFAULT_OUT = "data/eval/retrieval_golden.jsonl"


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN)
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT)

    if not src.exists():
        print(f"找不到勾选清单：{src}")
        return 1

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if r.get("_keep")]

    # ---------- 转换前校验 ----------
    problems: list[str] = []
    for r in kept:
        no = r.get("_no", "?")
        if not (r.get("question") or "").strip():
            problems.append(f"第 {no} 条：问题为空")
        if not (r.get("ground_truth") or "").strip():
            problems.append(f"第 {no} 条：标准答案为空")
        if not r.get("ground_truth_ids"):
            problems.append(f"第 {no} 条：没有标注 ground_truth_ids（检索指标会失真）")

    out = [
        {
            "question": r["question"].strip(),
            "ground_truth": r["ground_truth"].strip(),
            "ground_truth_ids": r["ground_truth_ids"],
        }
        for r in kept
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")

    # ---------- 汇总 ----------
    print(f"输入：{src}（{len(rows)} 条）")
    print(f"输出：{dst}（保留 {len(kept)} 条）")
    print()

    if kept:
        # 保留项里有多少是带标记的（提醒：人工判断，不是错误）
        flagged = [r for r in kept if r.get("_flags")]
        print(f"保留项中带机械标记的：{len(flagged)} 条（若是你有意保留，可忽略）")
        for r in flagged[:10]:
            print(f"  第 {r.get('_no')} 条：{'；'.join(r['_flags'])}")

        # 来源章节分布，帮你确认覆盖面
        dist = Counter((r.get("_heading") or "(无标题)")[:28] for r in kept)
        print(f"\n覆盖 {len(dist)} 个章节，分布前 10：")
        for h, n in dist.most_common(10):
            print(f"  {n:2} 条  {h}")

    if problems:
        print(f"\n⚠️ 发现 {len(problems)} 个问题，建议修好再跑评估：")
        for p in problems:
            print("  -", p)
        return 2

    print("\n✓ 格式校验通过。可以直接跑：")
    print(f'   POST /api/eval/retrieval  {{"dataset_path":"./{dst}","top_k":10,"run_name":"baseline-v1"}}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
