"""文档解析加载器：把 PDF / Word / HTML / TXT 统一转成 ParsedDocument。

统一约定：所有解析器都返回 `ParsedDocument`，上层（chunker）不关心来源格式，
替换解析库时只改本文件。

PDF 走三级阶梯（每级都做质量判定，不合格就降级）：
    ① pdfplumber（pdfminer）抽文字层 —— 常规 PDF 效果最好
    ② ①的字够多但**是乱码**时，换 PyMuPDF 重抽 —— 部分 CID 字体它能救回
    ③ 仍不合格或页均字符过少（扫描件）—— 降级 PaddleOCR

⚠️ 为什么必须有②和"乱码检测"：
国标/方正书版排版的 PDF（字体 FzBookMaker + Identity-H、缺 ToUnicode 映射）会出现
"文字层存在、但抽出来全是 `FGMÔTÕ5}` 这种乱码"的情况。只按字符数判断会把它
当正常文档入库，最终得到一库无法检索的垃圾。见 app/core/ingestion/quality.py。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.config import settings
from app.core.ingestion.quality import text_quality_score, top_offending_sample
from app.models.document import ParsedDocument
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 阈值集中由配置提供，便于按语料调整
MIN_TEXT_QUALITY = settings.min_text_quality

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".md",
                      ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def file_hash(path: str | Path) -> str:
    """文件内容 SHA-256，用于导入幂等（同一文件重复上传直接跳过）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- PDF 抽取阶梯
def _looks_two_column(page: Any) -> bool:
    """判断这一页是不是双栏排版。

    判据：**词块中心点落在中轴带内的比例**。
      · 单栏页面：每行横跨整页，中心点都在中轴附近 → 比例高 → 判单栏；
      · 双栏页面：词块宽度约等于一栏宽（实测消防法 ~220pt / 页宽 481pt），
        左右栏词块的中心分别落在两侧 → 中轴带内几乎没有词块 → 判双栏；
      · 例外：页眉、居中标题、修订日期这类横跨全页的行，中心点在中轴 —— 但它们行数少，
        用比例阈值即可容忍（消防法第 1 页正是这种情况）。

    为什么不用"投影分档看中轴低洼"：两栏间距只有约 30pt，比一个分档还窄，
    分档边界溢出后中轴并不出现低洼（实测比值 0.86~1.01，完全区分不出来）。
    """
    words = page.extract_words() or []
    if len(words) < 20:
        return False

    mid = page.width / 2
    band = page.width * 0.08          # 中轴 ±8% 视为"中轴带"
    centered = sum(1 for w in words if abs((w["x0"] + w["x1"]) / 2 - mid) <= band)
    return centered / len(words) < 0.15


def _table_to_markdown(rows: list[list[Any]]) -> str:
    """把 pdfplumber 抽出的表格转成 Markdown 表格。

    为什么必须转：GB 51283 里有 **29 个表格**，其中"防火间距"表是标准里最常被问的内容
    （甲类厂房与居住区的间距是多少米）。若按普通文本抽取，表格会被压平成断行碎片：

        居住区、村镇及重要公共建筑（建筑 |  | 90 | 100 | 140 | 50/60 | 60/70 | …

    行列对应关系丢失，模型无法判断"90 属于哪一列"，检索也几乎不可能命中。
    Markdown 表格保留行列结构，LLM 可以直接读表。
    """
    clean = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows]
    clean = [r for r in clean if any(r)]
    if not clean:
        return ""

    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    header, body = clean[0], clean[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _figure_marker(page: Any) -> str:
    """给含技术图的页面打占位标记。

    ⚠️ 当前技术栈**无法识别技术图的内容**（没有多模态模型）：
      · 图题（如"图 5.1.1 爆炸危险区域划分"）本身是文字，已被正常提取、可被检索；
      · 图形本身（示意线、区域划分、标注）无法转成文字。
    所以这里只插入标记，让切块与答案保留"此处有图"的信息，**不假装识别过**。
    真要理解图，需要接视觉模型（Qwen-VL / GPT-4o 等），属后续迭代。
    """
    try:
        images = page.images or []
    except Exception:  # noqa: BLE001
        return ""
    big = [im for im in images if (im.get("width", 0) or 0) * (im.get("height", 0) or 0) > 10000]
    return f"【本页含 {len(big)} 张技术图；图片内容未识别，图题见正文】" if big else ""


def _extract_body_text(page: Any) -> str:
    """抽正文文字：单栏直取，双栏按栏取（表格区域已在调用方剔除）。"""
    if not _looks_two_column(page):
        return page.extract_text() or ""

    words = page.extract_words() or []
    mid = page.width / 2
    # 顶部横跨两栏的区块（居中标题、修订日期）先整体抽出，
    # 否则会被中轴切成"中华人民共 / 和国消防法"这样的碎片
    spanning = [w for w in words if w["x0"] < mid < w["x1"]]
    top_span = [w for w in spanning if w["top"] < page.height * 0.35]

    head, y0 = "", 0.0
    if top_span:
        y0 = max(w["bottom"] for w in top_span) + 2
        head = page.crop((0, min(w["top"] for w in top_span) - 2, page.width, y0)).extract_text() or ""

    left = page.crop((0, y0, mid, page.height)).extract_text() or ""
    right = page.crop((mid, y0, page.width, page.height)).extract_text() or ""
    return "\n".join(part for part in (head, left, right) if part.strip())


def _extract_page_text(page: Any) -> str:
    """抽一页的文字；双栏页按"先左栏后右栏"分别抽取。

    为什么必须处理：pdfplumber 默认按 y 坐标逐行输出，**双栏页面会把左右两栏的句子
    交错拼进同一行**，例如消防法实测：

        位做好消防宣传教育工作。 第十一条 国务院住房和城乡建设主管部门
        教育、人力资源行政主管部门和学校、有关 规定的特殊建设工程，建设单位应当将消防设计

    这样入库后每个切块都是半句话拼接，语义破碎，检索和问答质量都会明显下降。
    """
    try:
        tables = page.find_tables()
    except Exception as exc:  # noqa: BLE001  个别页异常不应中断整篇
        logger.debug("表格检测失败：%s", exc)
        tables = []

    body_page = page
    table_blocks: list[str] = []
    for idx, table in enumerate(tables, 1):
        try:
            markdown = _table_to_markdown(table.extract())
        except Exception as exc:  # noqa: BLE001
            logger.debug("表格抽取失败：%s", exc)
            markdown = ""
        if markdown:
            table_blocks.append(f"【表格 {idx}】\n{markdown}")
        try:
            body_page = body_page.outside_bbox(table.bbox)   # 从正文里剔除表格区域，避免重复
        except Exception:  # noqa: BLE001
            pass

    parts = [_extract_body_text(body_page)]
    parts.extend(table_blocks)
    marker = _figure_marker(page)
    if marker:
        parts.append(marker)
    return "\n".join(p for p in parts if p and p.strip())


def _extract_with_pdfplumber(path: Path) -> tuple[str, int]:
    """① pdfplumber：常规 PDF 首选（表格能力也好）。返回 (全文, 页数)。"""
    import pdfplumber

    pages: list[str] = []
    two_column_pages = 0
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for idx, page in enumerate(pdf.pages):
            if _looks_two_column(page):
                two_column_pages += 1
            text = _extract_page_text(page)
            pages.append(f"\n【第 {idx + 1} 页】\n{text}")
            # TODO(第二批): 表格抽取用 page.extract_tables()，转成 Markdown 再入库
    if two_column_pages:
        logger.info("检测到 %s/%s 页为双栏排版，已按栏提取", two_column_pages, page_count)
    return "".join(pages), page_count


def _extract_with_pymupdf(path: Path) -> tuple[str, int]:
    """② PyMuPDF：处理 CID 字体的兜底抽取器。返回 (全文, 页数)。"""
    try:
        import pymupdf as fitz          # 新版包名
    except ImportError:                 # pragma: no cover
        import fitz                     # 旧版包名

    doc = fitz.open(str(path))
    try:
        pages = [
            f"\n【第 {i + 1} 页】\n" + doc.load_page(i).get_text("text")
            for i in range(doc.page_count)
        ]
        return "".join(pages), doc.page_count
    finally:
        doc.close()


def load_pdf(path: Path, use_ocr_fallback: bool = True) -> ParsedDocument:
    """PDF：文本层 -> 质量判定 -> (必要时换解析器) -> (必要时) OCR。"""
    text, page_count = _extract_with_pdfplumber(path)
    quality = text_quality_score(text)
    parser = "pdfplumber"

    # 页均字符太少 = 扫描件；字符够多但质量低 = 字体编码乱码。两者都换 PyMuPDF 试一次
    if quality < MIN_TEXT_QUALITY:
        logger.info("pdfplumber 抽取质量偏低（quality=%.3f），改用 PyMuPDF 重抽", quality)
        try:
            alt_text, alt_pages = _extract_with_pymupdf(path)
            alt_quality = text_quality_score(alt_text)
            if alt_quality > quality:
                text, page_count, quality, parser = alt_text, alt_pages, alt_quality, "pymupdf"
                logger.info("PyMuPDF 更优（quality=%.3f）", alt_quality)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyMuPDF 抽取失败，沿用 pdfplumber 结果: %s", exc)

    from app.core.ingestion.ocr import is_scanned_pdf, ocr_pdf

    too_few_chars = is_scanned_pdf(text, page_count)
    garbled = quality < MIN_TEXT_QUALITY
    if use_ocr_fallback and (too_few_chars or garbled):
        reason = "文本层字符过少（疑似扫描件）" if too_few_chars else f"文本层乱码（质量分 {quality:.3f}）"
        logger.warning("降级 PaddleOCR：%s；样例：%s", reason, top_offending_sample(text))
        ocr_text, ocr_pages = ocr_pdf(path, file_hash=file_hash(path))
        ocr_quality = text_quality_score(ocr_text)
        return ParsedDocument(
            title=path.stem,
            text=ocr_text,
            source_type="pdf",
            page_count=ocr_pages or page_count,
            is_scanned=True,
            metadata={
                "parser": "paddleocr",
                "ocr_reason": reason,
                "text_layer_quality": round(quality, 4),
                "text_layer_parser": parser,
                "ocr_quality": round(ocr_quality, 4),
                "original_chars": len(text),
            },
        )

    return ParsedDocument(
        title=path.stem,
        text=text,
        source_type="pdf",
        page_count=page_count,
        is_scanned=False,
        metadata={"parser": parser, "text_quality": round(quality, 4)},
    )


# ---------------------------------------------------------------- Word
def load_docx(path: Path) -> ParsedDocument:
    """Word：段落 + 表格，保留标题层级提示（Heading 样式转 Markdown ##）。"""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    lines: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if style.startswith("heading"):
            level = "".join(c for c in style if c.isdigit()) or "1"
            lines.append(f"\n{'#' * min(int(level), 6)} {text}")
        else:
            lines.append(text)
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))

    return ParsedDocument(
        title=path.stem,
        text="\n".join(lines),
        source_type="docx",
        page_count=0,
        metadata={"parser": "python-docx", "paragraphs": len(doc.paragraphs)},
    )


# ---------------------------------------------------------------- HTML
def load_html(path: Path | None = None, html: str = "", title: str = "") -> ParsedDocument:
    """HTML：用 BeautifulSoup 去脚本样式，保留标题结构。"""
    from bs4 import BeautifulSoup

    raw = html or Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()

    lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            lines.append(f"\n{'#' * int(el.name[1])} {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    page_title = title or (soup.title.get_text(strip=True) if soup.title else (path.stem if path else "untitled"))
    return ParsedDocument(
        title=page_title,
        text="\n".join(lines),
        source_type="html",
        metadata={"parser": "beautifulsoup4"},
    )


# ---------------------------------------------------------------- 纯文本
def load_text(path: Path) -> ParsedDocument:
    """TXT / Markdown 直读。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return ParsedDocument(
        title=path.stem,
        text=text,
        source_type="txt" if path.suffix.lower() == ".txt" else "md",
        metadata={"parser": "plain"},
    )


# ---------------------------------------------------------------- 统一入口
def load_file(path: str | Path, use_ocr_fallback: bool = True) -> ParsedDocument:
    """按后缀分发解析器；图片直接 OCR。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型: {suffix}")

    logger.info("开始解析 %s (%s)", p.name, suffix)
    if suffix == ".pdf":
        parsed = load_pdf(p, use_ocr_fallback=use_ocr_fallback)
    elif suffix in {".docx", ".doc"}:
        if suffix == ".doc":
            # TODO(第二批): 老 .doc 用 libreoffice --headless --convert-to docx 先转换
            raise ValueError(".doc 请先另存为 .docx（或用 libreoffice 转换）")
        parsed = load_docx(p)
    elif suffix in {".html", ".htm"}:
        parsed = load_html(path=p)
    elif suffix in {".txt", ".md"}:
        parsed = load_text(p)
    else:  # 图片
        from app.core.ingestion.ocr import ocr_file

        text, pages = ocr_file(p, file_hash=file_hash(p))
        parsed = ParsedDocument(title=p.stem, text=text, source_type="image",
                                page_count=pages, is_scanned=True, metadata={"parser": "paddleocr"})

    # 消防法规常见页眉页脚噪声，先做一次轻量清洗
    parsed.text = clean_text(parsed.text)
    logger.info("解析完成 %s：%s 字符，%s 页", p.name, len(parsed.text), parsed.page_count)
    return parsed


def normalize_width(text: str) -> str:
    """全角数字/字母 → 半角，并清理数字周围的多余空格。

    为什么需要（实测）：《中华人民共和国消防法》PDF 的文字层是全角的——

        （１９９８年４月２９日…根据２０２１年４月２９日…第二十八次会议）

    而用户提问一定用半角（"消防法第 28 条"）。不归一化的话：
      · BM25 精确匹配完全失效（`２８` ≠ `28`）—— 混合检索的一半能力被浪费；
      · embedding 也只能勉强匹配。

    ⚠️ 刻意**不动中文标点**（，。、；：（）《》）：它们是中文语料的正常组成部分，
    被我的文本质量分计入"有效字符"，改成半角反而会降低质量分并让文本不自然。
    只处理“数字/字母”这类会影响检索命中的字符。
    """
    # 1) 全角数字 ０-９、全角字母 Ａ-Ｚ ａ-ｚ、以及与数值相关的几个符号
    table = {}
    for i in range(10):
        table[0xFF10 + i] = chr(ord("0") + i)          # ０-９
    for i in range(26):
        table[0xFF21 + i] = chr(ord("A") + i)          # Ａ-Ｚ
        table[0xFF41 + i] = chr(ord("a") + i)          # ａ-ｚ
    table.update({
        0xFF0E: ".",    # ． 全角句点（数字里常见：３．５）
        0xFF0F: "/",    # ／ 标准号里常见：ＧＢ／Ｔ ３８３１５
        0xFF05: "%",    # ％
        0xFF0D: "-",    # －（范围/负号）
        0xFF0B: "+",    # ＋
        0xFF1D: "=",    # ＝
        0x3000: " ",    # 全角空格
        # 各种"连字符/破折号"统一成半角减号：标准号里 GB/T 38315—2019（em dash）与
        # 用户输入的 GB/T 38315-2019 必须能对上，否则 BM25 精确匹配会漏
        0x2013: "-", 0x2014: "-", 0x2015: "-", 0x2212: "-", 0xFF5E: "~", 0x301C: "~",
    })
    text = text.translate(table)

    # 2) 去掉“数字之间 / 汉字与数字之间 / 连字符与数字之间”的空格：
    #    "２０２１年４月２９日" 归一化后可能带空格 → "2021 年 4 月 29 日" → "2021年4月29日"
    #    PDF 抽出来的 "2020– 01–16"、"GB 51283 - 2020" 这类断字也会被修好
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=\d)\s+(?=-)", "", text)      # "51283 - 2020" → "51283- 2020"
    text = re.sub(r"(?<=-)\s+(?=\d)", "", text)      # "51283- 2020" → "51283-2020"
    return text


def clean_text(text: str) -> str:
    """清洗：全角归一化 → 去连续空行 → 去行尾空白。

    注意执行顺序：**先归一化再压缩空行**，否则"数字周围空格"的规则会看不到原始形态。
    TODO(第二批): 用规则/正则识别并剔除重复出现的页眉页脚与页码行。
    """
    text = normalize_width(text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        cleaned.append(ln)
    return "\n".join(cleaned).strip()
