"""拒答正确率测试：用负样本验证系统会不会"资料里没有却硬答"。

这是消防场景最关键的安全性指标，RAGAS 覆盖不到（faithfulness 只看答案是否被上下文支撑，
而"上下文本身不相关、模型却照着编"这种失败需要负样本来暴露）。

用法：
    cd backend
    python scripts/eval_abstention.py                      # 默认 data/eval/negative.jsonl
    python scripts/eval_abstention.py <文件> [--show-all]

判定规则（保守）：答案里出现任一"承认资料不足"的表述即算正确拒答；
若答案给出了**具体数值/要求**且没有任何不确定表述，则判为"硬答"（失败）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000/api/query"

# "承认资料不足"的表述——命中任一即视为正确拒答
REFUSE_MARKERS = (
    "未提供", "未收录", "未包含", "未涉及", "未给出", "未明确", "没有相关", "无相关规定",
    "无法确定", "无法回答", "不能确定", "资料中无", "资料中未", "现有资料", "本资料",
    "不在", "需查", "请查阅", "建议查阅", "超出", "不包含", "未找到",
)


def ask(question: str) -> dict:
    resp = httpx.post(
        API,
        json={"question": question, "top_k": 5, "use_cache": False},
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json().get("data", {})


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--show-all" in sys.argv
    path = Path(args[0] if args else "data/eval/negative.jsonl")

    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"负样本 {len(items)} 条（{path}）\n")

    rows = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        try:
            data = ask(q)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}] ⚠️ 调用失败：{type(exc).__name__}: {exc}")
            rows.append((i, q, False, "调用失败"))
            continue

        answer = (data.get("answer") or "").strip()
        refused = any(m in answer for m in REFUSE_MARKERS)
        verdict = "✅ 正确拒答" if refused else "❌ 硬答（可能编造）"
        rows.append((i, q, refused, data.get("latency_ms")))

        print("=" * 84)
        print(f"[{i}] {q}")
        print(f"    类别：{item.get('category', '')}")
        print(f"    预期：{item.get('acceptable', '说明资料中未提供')}")
        print(f"    判定：{verdict}   延迟 {data.get('latency_ms', 0) / 1000:.1f}s")
        head = answer.replace("\n", " ")
        print(f"    答案：{head[:230] if not show_all else head[:600]}")
        if not refused:
            print("    ⚠️ 没有出现任何'资料不足'表述 → 需要改用更严格的提示词或补充拒答规则")

    ok = sum(1 for r in rows if r[2])
    print("\n" + "=" * 84)
    print(f"拒答正确率：{ok}/{len(rows)} = {ok / max(len(rows), 1):.0%}（策略门槛 ≥90%）")
    if ok < len(rows):
        bad = [r[1] for r in rows if not r[2]]
        print("未正确拒答的问题：")
        for q in bad:
            print(f"  - {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
