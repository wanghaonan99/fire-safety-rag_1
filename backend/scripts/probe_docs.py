"""探查多个 PDF 的性质（只读，不写数据库、不加载模型）。

对每个文件给出：页数、文字层字符数、质量分、判定结果、文本样例。
用来决定"这份资料适合出哪类评估题"以及"是否需要 OCR"。

用法：cd backend && python scripts/probe_docs.py "路径1.pdf" "路径2.pdf" ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ingestion.quality import quality_breakdown, text_quality_score  # noqa: E402


def probe(path: Path) -> None:
    import pdfplumber

    print("=" * 78)
    print(f"文件：{path.name}")
    print(f"大小：{path.stat().st_size / 1024 / 1024:.2f} MB")

    pages_text: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        # 抽样前 8 页 + 中间 2 页，避免全量解析太慢
        sample_idx = list(range(min(8, page_count)))
        if page_count > 10:
            sample_idx.append(page_count // 2)
            sample_idx.append(page_count // 2 + 1)
        for i in sample_idx:
            pages_text.append(pdf.pages[i].extract_text() or "")

    sample = "\n".join(pages_text)
    chars_per_page = len(sample) / max(len(sample_idx), 1)
    q = quality_breakdown(sample)

    print(f"页数：{page_count}")
    print(f"抽样 {len(sample_idx)} 页，平均 {chars_per_page:.0f} 字符/页")
    print(f"质量分：{text_quality_score(sample):.3f}   (常用字 {q['cjk_common_ratio']:.3f} / "
          f"字母数字 {q['ascii_alnum_ratio']:.3f} / cid 标记 {int(q['has_cid_marker'])})")

    if q["has_cid_marker"]:
        verdict = "❌ 字体缺 ToUnicode（乱码），必须走 OCR"
    elif text_quality_score(sample) < 0.15:
        verdict = "❌ 文字层不可用（乱码或扫描件），需要 OCR"
    elif chars_per_page < 50:
        verdict = "❌ 基本没有文字层（扫描件），需要 OCR"
    else:
        verdict = "✅ 文字层正常，可直接解析"
    print(f"判定：{verdict}")

    preview = sample.strip().replace("\n", " ")[:220]
    print(f"样例：{preview}")
    print()


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("用法：python scripts/probe_docs.py <pdf> [pdf ...]")
        return 1
    for p in paths:
        if not p.exists():
            print(f"找不到文件：{p}\n")
            continue
        try:
            probe(p)
        except Exception as exc:  # noqa: BLE001
            print(f"探查失败：{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
