"""打印 PDF 指定页的原始文本行，用于判断是否存在"多栏交错"问题。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import pdfplumber

    path = Path(sys.argv[1])
    pages = [int(x) for x in sys.argv[2:]] or [2, 3]
    with pdfplumber.open(str(path)) as pdf:
        for pno in pages:
            if pno > len(pdf.pages):
                continue
            page = pdf.pages[pno - 1]
            print("=" * 88)
            print(f"第 {pno} 页（宽 {page.width:.0f} × 高 {page.height:.0f}）")
            print("-" * 88)
            text = page.extract_text() or ""
            for i, line in enumerate(text.splitlines()[:26], 1):
                print(f"{i:>3} | {line}")

            # 用词的 x 坐标判断是否多栏：统计每一行的词块横向分布
            words = page.extract_words()
            if words:
                xs = sorted({round(w["x0"] / 50) * 50 for w in words})
                print(f"\n  词块起始 x 坐标聚类（每 50pt 一档）: {xs}")
                print("  若 x 分布集中在两个相距很远的区间 → 双栏排版" if len(xs) > 6 else "")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
