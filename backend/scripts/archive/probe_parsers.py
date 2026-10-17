"""对比不同 PDF 解析器对同一文件的抽取效果，判断是否真的需要 OCR。"""

from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
TARGET_PAGE = 5   # 第 6 页，正文密度高

print("=" * 70)
print("① pdfplumber（当前 loader 用的，基于 pdfminer.six）")
try:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        text = pdf.pages[TARGET_PAGE].extract_text() or ""
    print(text[:400])
except Exception as exc:
    print("失败:", exc)

print()
print("=" * 70)
print("② PyMuPDF / fitz（不同的解析引擎）")
try:
    import pymupdf as fitz  # 新版包名

    doc = fitz.open(str(path))
    text = doc[TARGET_PAGE].get_text()
    doc.close()
    print(text[:400])
except ImportError:
    try:
        import fitz  # 旧版包名

        doc = fitz.open(str(path))
        text = doc[TARGET_PAGE].get_text()
        doc.close()
        print(text[:400])
    except ImportError:
        print("未安装（pip install pymupdf）")
except Exception as exc:
    print("失败:", exc)

print()
print("=" * 70)
print("③ pypdf")
try:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = reader.pages[TARGET_PAGE].extract_text() or ""
    print(text[:400])
except ImportError:
    print("未安装（pip install pypdf）")
except Exception as exc:
    print("失败:", exc)

print()
print("=" * 70)
print("④ 字体信息（判断是否缺 ToUnicode 映射）")
try:
    import pymupdf as fitz

    doc = fitz.open(str(path))
    page = doc[TARGET_PAGE]
    fonts = page.get_fonts(full=True)
    for f in fonts[:8]:
        # (xref, ext, type, basefont, name, encoding, ...)
        print(f"  xref={f[0]} type={f[2]} basefont={f[3]} encoding={f[5] if len(f) > 5 else '?'}")
    doc.close()
except Exception as exc:
    print("跳过:", exc)
