"""不需要 LLM 的 faithfulness 代理诊断。

思路：把答案按「结论 / 依据 / 操作要点」三段拆开，分别计算与检索上下文的
2-gram 重合度。若某段的平均重合度明显低于其他段，说明该段引入了上下文里没有的内容
——这正是 faithfulness 被拉低的位置。

另外统计"答案里出现、但上下文里从未出现的规范号/数值"（如 GB 50016），
这是模型注入外部知识的直接证据。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FILLED = Path("data/eval/runs/gen_filled.jsonl")

SECTION_PATTERNS = [
    ("结论", re.compile(r"结论[:：]\s*(.*?)(?=\n\s*[-*•]?\s*(?:依据|操作要点)[:：]|\Z)", re.S)),
    ("依据", re.compile(r"依据[:：]\s*(.*?)(?=\n\s*[-*•]?\s*(?:结论|操作要点)[:：]|\Z)", re.S)),
    ("操作要点", re.compile(r"操作要点[:：]\s*(.*)$", re.S)),
]

REG_PATTERN = re.compile(r"(?:GB|AQ|JGJ|SH|HG|DGJ|GA)\s*/?\s*T?\s*\d{3,5}(?:[-—.]\d{2,4})?")
NUM_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + 2] for i in range(max(len(s) - 1, 0))}


def overlap(part: str, context: str) -> float:
    a = bigrams(part)
    if not a:
        return 1.0
    return len(a & bigrams(context)) / len(a)


def main() -> int:
    rows = [json.loads(l) for l in FILLED.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"已答题 {len(rows)} 条\n")

    section_scores: dict[str, list[float]] = {name: [] for name, _ in SECTION_PATTERNS}
    missing_section = {name: 0 for name, _ in SECTION_PATTERNS}
    reg_leak = 0
    num_leak = 0
    examples: list[tuple[float, str, str]] = []

    for row in rows:
        answer = row.get("answer") or ""
        context = "\n".join(row.get("contexts") or [])
        if not context or not answer:
            continue

        parts: dict[str, str] = {}
        for name, pattern in SECTION_PATTERNS:
            m = pattern.search(answer)
            if m:
                parts[name] = m.group(1)
            else:
                missing_section[name] += 1

        for name, part in parts.items():
            if len(part.strip()) >= 8:
                section_scores[name].append(overlap(part, context))

        # 外部知识注入的直接证据
        ctx_regs = set(REG_PATTERN.findall(context))
        ans_regs = set(REG_PATTERN.findall(answer))
        if ans_regs - ctx_regs:
            reg_leak += 1
            if len(examples) < 6:
                examples.append((len(ans_regs - ctx_regs), row["question"][:52], ", ".join(sorted(ans_regs - ctx_regs))[:60]))

        ctx_nums = set(NUM_PATTERN.findall(context))
        ans_nums = {n for n in NUM_PATTERN.findall(answer) if len(n) >= 2}
        if ans_nums - ctx_nums:
            num_leak += 1

    print("各段与检索上下文的 2-gram 重合度（越低越可能是「无据内容」）：")
    for name, vals in section_scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            low = sum(1 for v in vals if v < 0.3)
            print(f"  {name:<8} n={len(vals):>4}  平均重合 {avg:.1%}   低于 30% 的 {low} 条")
        else:
            print(f"  {name:<8} （未出现该段）")

    print("\n外部知识注入的直接证据（答案里出现、上下文里没有的特征）：")
    print(f"  答案含上下文未出现的规范号：{reg_leak}/{len(rows)} 条（{reg_leak / max(len(rows), 1):.1%}）")
    print(f"  答案含上下文未出现的多位数值：{num_leak}/{len(rows)} 条（{num_leak / max(len(rows), 1):.1%}）")
    if examples:
        print("\n  规范号注入样例：")
        for _, q, regs in examples:
            print(f"    · {q}")
            print(f"      注入的规范号：{regs}")

    print("\n未出现某段的条数（说明模型没按三段式回答）：")
    for name, n in missing_section.items():
        print(f"  {name:<8} {n} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
