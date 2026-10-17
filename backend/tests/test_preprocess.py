"""预处理单测：全角归一化 + 低信息密度切块过滤。

覆盖两类"改了会直接影响评估结论"的处理：
  1. 全角数字/字母归一化 —— 不做的话 BM25 对"第28条"这类提问完全匹配不上；
  2. 低信息密度段落过滤 —— 不做的话封面/目次会被检索召回成噪声。
"""

from __future__ import annotations

from app.core.ingestion.chunker import is_low_value_chunk
from app.core.ingestion.loader import clean_text, normalize_width

# ---------------------------------------------------------------- 全角归一化


def test_全角数字转半角():
    """消防法 PDF 的真实形态（全角括号属于中文标点，按设计保留）。"""
    assert normalize_width("（１９９８年４月２９日）") == "（1998年4月29日）"
    assert normalize_width("第二十八次会议") == "第二十八次会议"


def test_全角字母与斜杠转半角():
    """标准号：ＧＢ／Ｔ ３８３１５ → GB/T 38315"""
    assert normalize_width("ＧＢ／Ｔ３８３１５") == "GB/T38315"
    assert normalize_width("ＡＱ／Ｔ ９００７") == "AQ/T 9007"


def test_全角句点与百分号():
    assert normalize_width("３．５") == "3.5"
    assert normalize_width("５０％") == "50%"


def test_破折号统一成减号():
    """标准号里的各种破折号 + 空格都要归一成用户会输入的形态。"""
    assert normalize_width("GB/T 38315—2019") == "GB/T 38315-2019"       # em dash
    assert normalize_width("GB 51283 – 2020") == "GB 51283-2020"        # en dash + 两侧空格
    assert normalize_width("300～500m") == "300~500m"                    # 全角波浪号


def test_数字周围空格被清除():
    """PDF 抽取常见 "１０月 ２８日" / "第 28 条" 这类断字。"""
    assert normalize_width("２００８年１０月 ２８日") == "2008年10月28日"
    assert normalize_width("第 28 条") == "第28条"
    assert normalize_width("2020– 01–16") == "2020-01-16"


def test_英文词组之间的空格保留():
    """不能把英文单词粘在一起（space 在字母之间要保留）。"""
    assert normalize_width("fire fighting standard") == "fire fighting standard"


def test_中文标点保持不变():
    """刻意不动中文标点：它们是中文语料的正常组成，改了反而降低质量分。"""
    src = "本标准适用于机关、团体、企业、事业单位（以下简称单位）。"
    assert normalize_width(src) == src


def test_clean_text_同时归一化并压缩空行():
    out = clean_text("第一行\u3000带全角空格\n\n\n\n第２８条　１２３项\n")
    assert "\u3000" not in out           # 全角空格已归一化
    assert "\n\n\n" not in out           # 连续空行已压缩
    assert "第28条123项" in out          # 全角数字半角化 + 汉字与数字间空格清除


# ---------------------------------------------------------------- 低信息密度过滤


TOC_SAMPLE = """目 次
1 总则 ........................................... 1
2 术语 ........................................... 3
3 火灾危险性分类 ................................. 8
4 厂址选择与工厂总平面布置 ...................... 12
5 工艺系统及生产设施 ............................ 20
"""

COVER_SAMPLE = """中华人民共和国国家标准
精细化工企业工程设计防火标准
2020-01-16 发布
2020-10-01 实施
中华人民共和国住房和城乡建设部 联合发布
"""

PREFACE_SAMPLE = """前 言
本标准是根据住房和城乡建设部《关于印发 2015 年工程建设标准规范制订、修订计划的通知》的要求编制的。
本标准主要起草人：邹某、李某、王某。
"""

REFERENCE_SAMPLE = """参考文献
[1] GB 50016—2014 建筑设计防火规范
[2] GB 50160—2008 石油化工企业设计防火规范
[3] GB/T 29639—2013 生产经营单位生产安全事故应急预案编制导则
"""

NORMAL_CLAUSE = """5.1.1 使用或生产甲、乙类物质的工艺系统设计，应符合下列规定：
1 宜采用排风系统；
2 应采取防止可燃气体、可燃液体泄漏的措施。
本条规定了工艺系统的防火要求。"""


def test_目次被判为低信息密度():
    ok, reason = is_low_value_chunk(TOC_SAMPLE, "目 次")
    assert ok and reason == "目次"


def test_封面被判为低信息密度():
    ok, reason = is_low_value_chunk(COVER_SAMPLE, "")
    assert ok and reason == "封面"


def test_前言与起草人被判为低信息密度():
    ok, reason = is_low_value_chunk(PREFACE_SAMPLE, "前 言")
    assert ok and reason == "前言/起草信息"


def test_参考文献被判为低信息密度():
    ok, reason = is_low_value_chunk(REFERENCE_SAMPLE, "参考文献")
    assert ok and reason == "参考文献"


def test_正常条款不会被误杀():
    """这是最重要的一条：过滤规则不能吃掉正文。"""
    ok, reason = is_low_value_chunk(NORMAL_CLAUSE, "5.1 一般规定")
    assert not ok, f"正文被误判为低信息密度：{reason}"


def test_含数字条款不被误杀():
    ok, _ = is_low_value_chunk("9.6.1 生产区等场所宜设置干粉型、水基型或泡沫型灭火器。", "9.6 灭火器设置")
    assert not ok


def test_空段落被判为低信息密度():
    ok, reason = is_low_value_chunk("   \n  ", "")
    assert ok and reason == "空段落"
