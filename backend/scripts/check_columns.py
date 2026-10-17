"""统计一个 PDF 里被判定为"双栏"的页数，用于验证版面判定不会误伤单栏文档。

用法：python scripts/check_columns.py <pdf> [pdf ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ingestion.loader import _looks_two_column  # noqa: E402


def main() -> int:
    import pdfplumber

    for raw in sys.argv[1:]:
        p = Path(raw)
        if not p.exists():
            print(f"找不到：{p}")
            continue
        with pdfplumber.open(str(p)) as pdf:
            verdicts = [_looks_two_column(pg) for pg in pdf.pages]
        hit = [i + 1 for i, v in enumerate(verdicts) if v]
        print("=" * 78)
        print(f"{p.name}   共 {len(verdicts)} 页")
        print(f"  判定为双栏的页: {hit if hit else '（无）'}")
        print(f"  占比 {len(hit) / max(len(verdicts), 1):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
