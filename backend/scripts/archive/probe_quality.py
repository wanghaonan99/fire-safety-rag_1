"""验证乱码检测公式：真实 PDF + 正常中文/英文对照。"""

from __future__ import annotations

import sys
from pathlib import Path

from app.core.ingestion.loader import _extract_with_pdfplumber, _extract_with_pymupdf
from app.core.ingestion.quality import MIN_TEXT_QUALITY, quality_breakdown

path = Path(sys.argv[1])

NORMAL_CN = (
    "本标准规定了社会单位灭火和应急疏散预案的编制要求。单位应当根据本单位火灾危险性，"
    "确定消防安全重点部位，明确各岗位人员的职责分工，并定期组织演练。灭火和应急疏散预案应当包括"
    "组织机构、报警和接警处置程序、应急疏散的组织程序和措施、扑救初起火灾的程序和措施。"
)
NORMAL_EN = (
    "The standard specifies requirements for fire safety plans. Units shall organize "
    "drills periodically and review the plan at least once a year."
)

pp, pages = _extract_with_pdfplumber(path)
mu, _ = _extract_with_pymupdf(path)

cases = [
    ("正常中文", NORMAL_CN, "正常"),
    ("正常英文", NORMAL_EN, "正常"),
    ("空文本", "", "乱码"),
    ("本文件-pdfplumber", pp, "乱码"),
    ("本文件-PyMuPDF", mu, "乱码"),
]

print(f"判定阈值 MIN_TEXT_QUALITY = {MIN_TEXT_QUALITY}\n")
print(f"{'样本':<20}{'质量分':>8}{'常用字':>9}{'字母数字':>10}{'空格率':>8}{'cid':>6}  {'判定':<12}{'期望'}")
print("-" * 90)

ok = True
for label, text, expect in cases:
    b = quality_breakdown(text)
    verdict = "乱码" if b["quality"] < MIN_TEXT_QUALITY else "正常"
    flag = "✓" if verdict == expect else "✗ 不符"
    ok = ok and verdict == expect
    print(
        f"{label:<20}{b['quality']:>8.4f}{b['cjk_common_ratio']:>9.4f}{b['ascii_alnum_ratio']:>10.4f}"
        f"{b['space_ratio']:>8.4f}{int(b['has_cid_marker']):>6}  {verdict:<12}{expect} {flag}"
    )

print("-" * 90)
print("全部符合预期 ✓" if ok else "存在不符项 ✗")
