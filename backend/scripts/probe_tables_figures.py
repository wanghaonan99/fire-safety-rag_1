"""统计 PDF 里的表格与图片数量，并抽样看表格抽取质量。

用法：python scripts/probe_tables_figures.py <pdf> [pdf ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def probe(path: Path) -> None:
    import pdfplumber
    import pymupdf

    print("=" * 84)
    print(f"📄 {path.name}")

    table_pages: list[int] = []
    table_count = 0
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            try:
                tables = page.find_tables()
            except Exception:  # noqa: BLE001
                tables = []
            if tables:
                table_pages.append(i)
                table_count += len(tables)

    doc = pymupdf.open(str(path))
    image_counts = {}
    for i in range(doc.page_count):
        imgs = doc.load_page(i).get_images(full=True)
        if imgs:
            image_counts[i + 1] = len(imgs)
    doc.close()

    print(f"  表格：{table_count} 个，分布在 {len(table_pages)} 页"
          f"{'（页: ' + ','.join(map(str, table_pages[:20])) + ('…' if len(table_pages) > 20 else '') + '）' if table_pages else ''}")
    print(f"  图片：{sum(image_counts.values())} 张，分布在 {len(image_counts)} 页"
          f"{'（页: ' + ','.join(map(str, list(image_counts)[:20])) + ('…' if len(image_counts) > 20 else '') + '）' if image_counts else ''}")

    # 抽样第一个表格，看抽取质量
    if table_pages:
        pno = table_pages[0]
        with pdfplumber.open(str(path)) as pdf:
            tables = pdf.pages[pno - 1].find_tables()
            if tables:
                t = tables[0]
                print(f"\n  第一个表格（第 {pno} 页）· {len(t.rows)} 行 × {len(t.rows[0].cells) if t.rows else 0} 列：")
                for row in t.extract()[:6]:
                    cells = [(c or "").replace("\n", " ")[:16] for c in row]
                    print("    | " + " | ".join(cells) + " |")
    print()


def main() -> int:
    for raw in sys.argv[1:]:
        p = Path(raw)
        if p.exists():
            try:
                probe(p)
            except Exception as exc:  # noqa: BLE001
                print(f"  探查失败：{type(exc).__name__}: {exc}\n")
        else:
            print(f"找不到：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
