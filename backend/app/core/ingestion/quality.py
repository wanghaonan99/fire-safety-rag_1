"""文本质量判定：识别"有文字层但抽出来是乱码"的 PDF。

背景（为什么需要这个文件）：
国标 / 方正书版排版的 PDF 常用 `FzBookMaker` 系列字体 + `Identity-H` 编码，
且**不带 ToUnicode 映射表**。这类 PDF 的文字层"存在"、字符数也很多，但抽出来是：

    pdfplumber : 犌犅／犜３８３１５—２０１９ \n FGM(cid:212)T(cid:213)5}(cid:204)!FG(M}`O
    PyMuPDF    : ａ）　FGMÔTÕ5}Ì!FG(M}ÁO，"a®Ü8G&，tÚ３００ｍ～５００ｍ01(UY

如果只按"字符数够多 = 有文字层"判断，这种文件会被当正常文档入库，
最终得到一库无法检索、且**不报错**的垃圾。

判定用的三个信号（实测标定见下表）：
    信号            正常中文   正常英文  乱码(pdfminer)  乱码(PyMuPDF)
    常用字覆盖率      0.325      0.000      0.0004         0.0013
    ASCII 字母数字率  0.000      0.983      0.577          0.231
    空格率            0.000      0.149      0.036          0.009
    (cid: 标记        无         无         大量           无

结论：常用字覆盖率是最强判别器（差两个数量级）；
"(cid:" 是 pdfminer 遇到缺 ToUnicode 字体时的确定性标志；
真英文文档的字母数字率高**且**空格率高，乱码则空格率极低。
"""

from __future__ import annotations

# 中文最高频的 200 余字：正常行文里占比通常 0.25~0.40
COMMON_CJK = set(
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家"
    "可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情"
    "己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身"
    "者被高已亲其进此话常与活正感必须应安全消防求要求规定标准本项内容按执行"
)

# 中文标点（正常中文文档里占比约 0.07~0.15）
CJK_PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·％℃")

# ============ 阈值（与 config.Settings.min_text_quality 对应）============
MIN_TEXT_QUALITY = 0.15      # 质量分低于它 -> 判定乱码 -> 换解析器或 OCR

# 标定常数
CJK_COVERAGE_FULL = 0.25     # 常用字覆盖率达到这个值即算满分
ASCII_ALNUM_FULL = 0.70      # 字母数字率达到这个值即算满分（英文文档）
MIN_SPACE_RATIO = 0.08       # 真英文文档的空格率下界（乱码通常 < 0.04）

# 确定性乱码标志：pdfminer 拿不到 Unicode 时会直接输出 (cid:NNN)
CID_MARKER = "(cid:"

_SAMPLE_CHARS = 20000        # 只抽样前 2 万字，够判断且快


def text_quality_score(text: str) -> float:
    """返回 0~1 的文本质量分。参考区间（实测）：

    - 正常中文法规文档：0.9 ~ 1.0
    - 正常英文文档    ：0.9 ~ 1.0
    - 方正书版乱码 PDF：< 0.01
    - 纯扫描件（无文字层）：0.0
    """
    sample = text[:_SAMPLE_CHARS]
    if not sample.strip():
        return 0.0

    # 遇到大量 CID 编号，直接判定不可用（无需再算比例）
    if CID_MARKER in sample:
        return 0.0

    compact = "".join(sample.split())      # 去空白后算字符类别占比
    n = len(compact) or 1

    cjk_common = sum(1 for ch in compact if ch in COMMON_CJK) / n
    ascii_alnum = sum(1 for ch in compact if ch.isascii() and ch.isalnum()) / n
    space_ratio = sample.count(" ") / len(sample)

    chinese_score = min(cjk_common / CJK_COVERAGE_FULL, 1.0)
    # 英文只有在空格率合理（说明是真词句而非乱码字母堆）时才给分
    english_score = (
        min(ascii_alnum / ASCII_ALNUM_FULL, 1.0) if space_ratio >= MIN_SPACE_RATIO else 0.0
    )
    return max(chinese_score, english_score)


def is_garbled_text(text: str, threshold: float = MIN_TEXT_QUALITY) -> bool:
    """文本质量分低于阈值即认为是乱码（需要换解析器或改用 OCR）。"""
    return text_quality_score(text) < threshold


def quality_breakdown(text: str) -> dict[str, float]:
    """拆出各项指标，便于排查与调参。"""
    sample = text[:_SAMPLE_CHARS]
    compact = "".join(sample.split())
    n = len(compact) or 1
    return {
        "quality": round(text_quality_score(text), 4),
        "cjk_common_ratio": round(sum(1 for c in compact if c in COMMON_CJK) / n, 4),
        "ascii_alnum_ratio": round(sum(1 for c in compact if c.isascii() and c.isalnum()) / n, 4),
        "space_ratio": round(sample.count(" ") / max(len(sample), 1), 4),
        "has_cid_marker": float(CID_MARKER in sample),
        "chars": float(len(text)),
    }


def top_offending_sample(text: str, length: int = 120) -> str:
    """取一段"最像乱码"的样本文本，写进日志便于人工确认。"""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    if not lines:
        return text[:length].replace("\n", " ")
    worst = min(lines[:50], key=text_quality_score)
    return worst[:length]
