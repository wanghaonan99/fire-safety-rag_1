"""探测一个 PDF：页数、每页字符数、是否扫描件（不加载任何模型）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber

from app.core.ingestion.ocr import MIN_CHARS_PER_PAGE, is_scanned_pdf

path = Path(sys.argv[1])
print("文件      :", path.name)
print("大小      :", f"{path.stat().st_size / 1024 / 1024:.2f} MB")

with pdfplumber.open(str(path)) as pdf:
    pages = len(pdf.pages)
    chars = [len((p.extract_text() or "").strip()) for p in pdf.pages[:10]]

print("页数      :", pages)
print("前 10 页字符数:", chars)
avg = sum(chars) / max(len(chars), 1)
print("前 10 页平均字符数/页:", round(avg, 1), f"（判定阈值 {MIN_CHARS_PER_PAGE}）")
print("是否扫描件:", is_scanned_pdf("x" * int(avg), 1) if False else (avg < MIN_CHARS_PER_PAGE))
print()
print("→ 判定：", "扫描件，须走 PaddleOCR（慢）" if avg < MIN_CHARS_PER_PAGE else "有文字层，直接 pdfplumber 解析（快）")

with pdfplumber.open(str(path)) as pdf:
    sample = (pdf.pages[1].extract_text() or "")[:300] if pages > 1 else ""
print("\n--- 第 2 页文本样例 ---")
print(sample or "(空)")
