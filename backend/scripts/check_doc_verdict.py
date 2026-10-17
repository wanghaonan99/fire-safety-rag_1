"""对指定的 PDF 跑一遍"loader 会怎么判定"，确认会不会被误送去 OCR。

用法：python scripts/check_doc_verdict.py <pdf> [pdf ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ingestion.loader import _extract_with_pdfplumber  # noqa: E402
from app.core.ingestion.ocr import is_scanned_pdf  # noqa: E402
from app.core.ingestion.quality import MIN_TEXT_QUALITY, quality_breakdown  # noqa: E402


def main() -> int:
    for raw in sys.argv[1:]:
        p = Path(raw)
        if not p.exists():
            print(f"找不到：{p}")
            continue
        text, pages = _extract_with_pdfplumber(p)
        q = quality_breakdown(text)
        scanned = is_scanned_pdf(text, pages)
        verdict = "❌ 触发 OCR" if (scanned or q["quality"] < MIN_TEXT_QUALITY) else "✅ 直接用文字层"
        print("=" * 78)
        print(f"{p.name}")
        print(f"  页数 {pages} · 字符数 {int(q['chars'])} · 每页 {int(q['chars']) / max(pages, 1):.0f} 字")
        print(f"  质量分 {q['quality']:.3f}（阈值 {MIN_TEXT_QUALITY}）"
              f" · 常用字 {q['cjk_common_ratio']:.3f} · 字母数字 {q['ascii_alnum_ratio']:.3f}"
              f" · cid 标记 {int(q['has_cid_marker'])}")
        print(f"  页均字符判定（扫描件）: {scanned}")
        print(f"  → {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
