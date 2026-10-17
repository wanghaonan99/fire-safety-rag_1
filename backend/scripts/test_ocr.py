"""验证 OCR 链路（只跑前 N 页），检查 PaddleOCR 版本适配与识别质量。"""

from __future__ import annotations

import sys
from pathlib import Path

from app.core.ingestion.ocr import ocr_pdf, paddleocr_major_version
from app.core.ingestion.quality import quality_breakdown, text_quality_score

path = Path(sys.argv[1])
pages_to_run = int(sys.argv[2]) if len(sys.argv) > 2 else 2

print("PaddleOCR 主版本:", paddleocr_major_version())
print(f"开始 OCR 前 {pages_to_run} 页（首次会下载检测/识别模型）...")

text, pages = ocr_pdf(path, file_hash="probe-ocr", max_pages=pages_to_run)

print(f"\n完成：{pages} 页，{len(text)} 字符")
print("质量拆解:", quality_breakdown(text))
print("质量分:", round(text_quality_score(text), 4))
print("\n--- 识别文本前 600 字 ---")
print(text[:600])
