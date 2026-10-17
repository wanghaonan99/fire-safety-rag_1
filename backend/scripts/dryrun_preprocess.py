"""预处理干跑验证：在真实 PDF 上看"全角归一化"和"低信息密度过滤"的效果。

**不写数据库、不加载模型**（语义切块需要模型，这里只做段落级验证）。

用法：cd backend && python scripts/dryrun_preprocess.py <pdf> [pdf ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ingestion.chunker import is_low_value_chunk, split_by_heading  # noqa: E402
from app.core.ingestion.loader import clean_text, load_file  # noqa: E402

FULLWIDTH_DIGIT = re.compile(r"[\uff10-\uff19]")


def show_normalization(path: Path) -> None:
    """对比归一化前后：找一行含全角数字的原文。"""
    from app.core.ingestion.loader import _extract_with_pdfplumber

    raw, _ = _extract_with_pdfplumber(path)
    hits = [ln.strip() for ln in raw.splitlines() if FULLWIDTH_DIGIT.search(ln)]
    if not hits:
        print("  （本文档没有全角数字，无需归一化）")
        return
    before = hits[0][:96]
    after = clean_text(before)[:96]
    print(f"  归一化前：{before}")
    print(f"  归一化后：{after}")


def main() -> int:
    for raw_path in sys.argv[1:]:
        p = Path(raw_path)
        if not p.exists():
            print(f"找不到：{p}")
            continue

        print("=" * 90)
        print(f"📄 {p.name}")
        print("-" * 90)
        print("[1] 全角归一化")
        show_normalization(p)

        print("\n[2] 低信息密度过滤")
        parsed = load_file(p)                       # 解析 + clean_text（不需要模型）
        sections = split_by_heading(parsed.text) or [("", parsed.text)]
        dropped: list[tuple[str, str, str]] = []
        kept: list[tuple[str, str]] = []
        for heading, body in sections:
            low, reason = is_low_value_chunk(body, heading)
            (dropped if low else kept).append((heading, reason if low else "", body))

        print(f"  段落总数 {len(sections)} → 保留 {len(kept)}，丢弃 {len(dropped)}")
        if dropped:
            print("  丢弃明细：")
            for heading, reason, body in dropped:
                preview = body.strip().replace("\n", " ")[:60]
                print(f"    · [{reason}] {heading[:18] or '(无标题)'} | {preview}…")
        if kept:
            print("  保留样例（前 2 条）：")
            for heading, _, body in kept[:2]:
                print(f"    · {heading[:24] or '(无标题)'} | {body.strip()[:60]}…")

        print(f"\n[3] 结果：{len(kept)} 个段落将进入语义切块（本脚本不执行切块/向量化）")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
