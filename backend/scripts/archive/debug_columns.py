"""调试双栏投影：打印每页的横向覆盖分布。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import pdfplumber

    path = Path(sys.argv[1])
    pages = [int(x) for x in sys.argv[2:]] or [1, 2]
    with pdfplumber.open(str(path)) as pdf:
        for pno in pages:
            page = pdf.pages[pno - 1]
            words = page.extract_words() or []
            bins = 20
            mass = [0] * bins
            for w in words:
                lo = max(int(w["x0"] / page.width * bins), 0)
                hi = min(int(w["x1"] / page.width * bins), bins - 1)
                for b in range(lo, hi + 1):
                    mass[b] += 1
            mid = bins // 2
            mid_mass = sum(mass[mid - 1 : mid + 2]) / 3
            side_mass = max(min(mass[2], mass[-3]), 1)
            print(f"第 {pno} 页  宽 {page.width:.0f}pt  词块 {len(words)} 个")
            print(f"  覆盖分布: {mass}")
            print(f"  中轴均值={mid_mass:.1f}  两侧参考={side_mass}  比值={mid_mass / side_mass:.2f}")
            print(f"  当前判定: {'双栏' if mid_mass < 0.4 * side_mass else '单栏'}")
            # 词块宽度统计，看是否被合并
            widths = sorted(round(w["x1"] - w["x0"]) for w in words)[:5] + sorted(round(w["x1"] - w["x0"]) for w in words)[-5:]
            print(f"  词块宽度样本: {widths}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
