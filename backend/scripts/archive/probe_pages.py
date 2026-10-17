"""逐页探查一个 PDF 的文字层质量，定位"哪几页是乱码/扫描件"。

用法：python scripts/probe_pages.py <pdf> [页码...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ingestion.quality import text_quality_score  # noqa: E402


def main() -> int:
    import pdfplumber

    path = Path(sys.argv[1])
    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        idxs = [int(x) for x in sys.argv[2:]] or list(range(0, total, max(total // 12, 1)))

        print(f"文件：{path.name}   共 {total} 页")
        print(f"{'页码':>5}  {'字符数':>7}  {'质量分':>7}  判定 / 样例")
        print("-" * 96)
        bad = 0
        for i in idxs:
            if i >= total:
                continue
            text = pdf.pages[i].extract_text() or ""
            score = text_quality_score(text)
            if score < 0.15:
                verdict, bad = "❌ 乱码/无文字层", bad + 1
            elif score < 0.4:
                verdict = "⚠️ 质量偏低"
            else:
                verdict = "✅ 正常"
            preview = text.strip().replace("\n", " ")[:52]
            print(f"{i + 1:>5}  {len(text):>7}  {score:>7.3f}  {verdict}  {preview}")
        print("-" * 96)
        print(f"抽样 {len(idxs)} 页中 {bad} 页不可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
