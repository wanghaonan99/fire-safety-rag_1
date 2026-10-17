"""乱码检测的单元测试。

用真实遇到的乱码样本做回归保护：这类 bug 一旦退回，结果是"库能建起来、
但检索全是垃圾，且不报错"，非常隐蔽。
"""

from __future__ import annotations

from app.core.ingestion.quality import (
    MIN_TEXT_QUALITY,
    is_garbled_text,
    quality_breakdown,
    text_quality_score,
)

# 真实样本：GB/T 38315-2019（方正书版 FzBookMaker 字体，无 ToUnicode 映射）
GARBLED_PDFMINER = (
    "犌犅／犜３８３１５—２０１９\n"
    "FGM(cid:212)T(cid:213)5}(cid:204)!FG(M}`O \"afi(cid:220)8G& t(cid:218) 01(cid:139)(UY\n"
    "(cid:210)(cid:211) <=Æ(cid:130) <(cid:132)(cid:226)‚`(cid:150)(cid:155)Y(cid:128)9G(T,G&"
)
GARBLED_PYMUPDF = (
    "ａ）　FGMÔTÕ5}Ì!FG(M}ÁO，\"a®Ü8G&，tÚ３００ｍ～５００ｍ01(UY\n"
    "XY、Û2<=Ð、Ü<=Ý、./5=ÍÎÒÓÏa，9XYXY、ÐÞ、¹ß(à"
)
NORMAL_CN = (
    "本标准规定了社会单位灭火和应急疏散预案的编制要求。单位应当根据本单位火灾危险性，"
    "确定消防安全重点部位，明确各岗位人员的职责分工，并定期组织演练。预案应当包括"
    "组织机构、报警和接警处置程序、应急疏散的组织程序和措施、扑救初起火灾的程序和措施。"
)
NORMAL_EN = (
    "The standard specifies requirements for fire safety plans. Units shall organize "
    "drills periodically and review the plan at least once a year."
)


def test_正常中文不算乱码():
    assert text_quality_score(NORMAL_CN) >= MIN_TEXT_QUALITY
    assert not is_garbled_text(NORMAL_CN)


def test_正常英文不算乱码():
    assert text_quality_score(NORMAL_EN) >= MIN_TEXT_QUALITY
    assert not is_garbled_text(NORMAL_EN)


def test_空文本算乱码():
    assert text_quality_score("") == 0.0
    assert is_garbled_text("")


def test_cid标记直接判为乱码():
    """pdfminer 拿不到 Unicode 时会输出 (cid:NNN)，这是确定性标志。"""
    assert text_quality_score(GARBLED_PDFMINER) == 0.0
    assert is_garbled_text(GARBLED_PDFMINER)
    assert quality_breakdown(GARBLED_PDFMINER)["has_cid_marker"] == 1.0


def test_方正字体乱码判为乱码():
    """PyMuPDF 抽出来没有 (cid: 标记，但常用字覆盖率接近 0，必须也能识别。"""
    score = text_quality_score(GARBLED_PYMUPDF)
    assert score < MIN_TEXT_QUALITY, f"质量分 {score} 应低于阈值"
    assert is_garbled_text(GARBLED_PYMUPDF)
    assert quality_breakdown(GARBLED_PYMUPDF)["cjk_common_ratio"] < 0.01


def test_正常中文常用字覆盖率显著高于乱码():
    normal = quality_breakdown(NORMAL_CN)["cjk_common_ratio"]
    garbled = quality_breakdown(GARBLED_PYMUPDF)["cjk_common_ratio"]
    assert normal > 0.2
    assert normal > garbled * 50      # 差两个数量级才可靠


def test_字母堆但不含空格的英文不算正常():
    """乱码的一种形态：拉丁字母很多但没有词间空格，不能当英文文档放行。"""
    letters_only = "FGMÔTÕ5}ÌFGMÁOaÜ8GtÚ01UYXYÛ2ÍÎÒÓÏa9XYXYÐÞßà"
    assert is_garbled_text(letters_only)
