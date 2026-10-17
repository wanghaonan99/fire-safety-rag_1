"""OCR 降级模块（第二批核心能力之一）。

策略：
1. 图片型/扫描型 PDF，或**文字层乱码**的 PDF（见 loader.py 的判定）-> 走 PaddleOCR；
2. 渲染 PDF 每页为图片用 PyMuPDF(fitz)（比 pdf2image 少一个 poppler 依赖）；
3. OCR 结果按页缓存到 data/indexes/ocr_cache/<file_hash>/page_<n>.txt，
   重复导入同一文件时直接命中，省掉最贵的一步。

⚠️ PaddleOCR 2.x 与 3.x 的 API 完全不同（3.x 移除了 use_angle_cls/show_log，
   返回值也从嵌套 list 变成 dict-like 的 OCRResult）。本文件按主版本号自动适配，
   避免"装到 3.x 就报 unknown argument"这类问题。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

_OCR_ENGINE: Any = None
_OCR_MAJOR: int = 0          # 2 或 3，决定用哪套调用与解析方式
OCR_CACHE_DIR = Path("./data/indexes/ocr_cache")

# 判定"是扫描件"的阈值：平均每页有效字符数低于该值即认为没有文字层
MIN_CHARS_PER_PAGE = 50


def paddleocr_major_version() -> int:
    """返回已安装 PaddleOCR 的主版本号（拿不到就返回 0）。"""
    try:
        import paddleocr

        return int(str(getattr(paddleocr, "__version__", "0")).split(".")[0])
    except Exception:  # noqa: BLE001
        return 0


def _create_engine() -> None:
    """懒加载 PaddleOCR（首次导入会下载约 10-30MB 检测/识别模型）。"""
    global _OCR_ENGINE, _OCR_MAJOR
    if _OCR_ENGINE is not None:
        return

    from paddleocr import PaddleOCR

    _OCR_MAJOR = paddleocr_major_version()
    if _OCR_MAJOR >= 3:
        # 3.x：角度分类改名为 textline_orientation；文档方向/去扭曲单独开关
        #
        # 三个开关的取舍（都是实测结论）：
        #   use_textline_orientation  文字行方向判断。对正立的印刷文档收益极低，
        #                             且它依赖的 PP-LCNet_x1_0_textline_ori 模型
        #                             在国内网络下常常下载失败（目录建了但没文件），
        #                             因此默认关闭；倒置/竖排扫描件再打开。
        #   use_doc_orientation_classify  整页旋转判断，单页扫描件不需要
        #   use_doc_unwarping             文档去扭曲，对标准印刷件耗时高、收益低
        textline_ori = os.getenv("OCR_TEXTLINE_ORIENTATION", "false").lower() in {"1", "true", "yes"}
        logger.info("初始化 PaddleOCR %s.x 引擎（textline_orientation=%s）", _OCR_MAJOR, textline_ori)
        _OCR_ENGINE = PaddleOCR(
            lang="ch",
            use_textline_orientation=textline_ori,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    else:
        logger.info("初始化 PaddleOCR %s.x 引擎", _OCR_MAJOR)
        _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)


def _parse_result(res: Any) -> list[str]:
    """把引擎返回值解析成文本行，兼容 2.x / 3.x 两种结构。"""
    texts: list[str] = []

    if _OCR_MAJOR >= 3:
        # 3.x：OCRResult（dict 子类），键为 rec_texts / rec_scores / rec_polys
        data: Any = res
        if not hasattr(data, "get"):
            data = getattr(res, "json", {})
            data = data.get("res", data) if hasattr(data, "get") else {}
        rec_texts = data.get("rec_texts") if hasattr(data, "get") else None
        if rec_texts:
            texts.extend(str(t) for t in rec_texts)
        return texts

    # 2.x：[[ [box, (text, score)], ... ], ...]  (按页分组)
    for page in res or []:
        for item in page or []:
            try:
                text, _score = item[1]
                texts.append(str(text))
            except (IndexError, TypeError):
                continue
    return texts


def _run_ocr(engine: Any, image_path: str | Path) -> list[str]:
    """调用引擎，兼容 3.x 的 predict() 与 2.x 的 ocr()。"""
    if _OCR_MAJOR >= 3 and hasattr(engine, "predict"):
        result = engine.predict(str(image_path))
    elif _OCR_MAJOR >= 3:
        result = engine.ocr(str(image_path))
    else:
        result = engine.ocr(str(image_path), cls=True)

    lines: list[str] = []
    for res in result or []:
        lines.extend(_parse_result(res))
    return lines


def is_scanned_pdf(text: str, page_count: int) -> bool:
    """判断是否扫描件：文字层字符数 / 页数 < 阈值。

    注意：这只是"字符太少"的判定；"字符够多但是乱码"由
    app/core/ingestion/quality.py 的质量分负责，两者在 loader 里合并判断。
    """
    if page_count <= 0:
        return len(text.strip()) < MIN_CHARS_PER_PAGE
    return (len(text.strip()) / max(page_count, 1)) < MIN_CHARS_PER_PAGE


def _cache_dir(file_hash: str) -> Path:
    d = OCR_CACHE_DIR / file_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def ocr_image(image_path: str | Path, file_hash: str = "", page_no: int = 0) -> str:
    """对单张图片做 OCR，返回拼接后的文本（按页磁盘缓存）。"""
    if file_hash:
        cache_file = _cache_dir(file_hash) / f"page_{page_no}.txt"
        if cache_file.exists():
            logger.debug("OCR 缓存命中 page=%s", page_no)
            return cache_file.read_text(encoding="utf-8")

    _create_engine()
    lines = _run_ocr(_OCR_ENGINE, image_path)
    content = "\n".join(lines)

    if file_hash:
        (_cache_dir(file_hash) / f"page_{page_no}.txt").write_text(content, encoding="utf-8")
    return content


def ocr_pdf(pdf_path: str | Path, file_hash: str = "", max_pages: int = 0) -> tuple[str, int]:
    """整份扫描件 PDF 逐页 OCR。返回 (全文, 页数)。

    TODO(第二批):
      - 并行化：多页 OCR 用 concurrent.futures 提速度（注意 Paddle 线程安全）；
      - 版面分析：表格/多栏用 PP-StructureV3 或 MinerU 还原阅读顺序；
      - 纠错：法规编号（GB 50016-2014）等专有名词用正则做后处理。
    """
    try:
        import pymupdf as fitz          # 新版包名
    except ImportError as exc:          # pragma: no cover
        try:
            import fitz                 # 旧版包名
        except ImportError:
            raise ImportError(
                "OCR 需要 PyMuPDF 把 PDF 渲染成图片，请先安装：pip install pymupdf"
            ) from exc

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count if max_pages <= 0 else min(max_pages, doc.page_count)
    texts: list[str] = []
    tmp_dir = _cache_dir(file_hash or hashlib.md5(str(pdf_path).encode()).hexdigest())

    for i in range(page_count):
        page = doc.load_page(i)
        # zoom=2.8 约等于 200 DPI，再高收益递减、耗时翻倍
        pix = page.get_pixmap(matrix=fitz.Matrix(2.8, 2.8))
        img_path = tmp_dir / f"raw_page_{i}.png"
        pix.save(str(img_path))
        texts.append(f"\n【第 {i + 1} 页】\n" + ocr_image(img_path, file_hash=file_hash, page_no=i))
        logger.info("OCR 进度 %s/%s", i + 1, page_count)

    doc.close()
    return "".join(texts), page_count


def ocr_file(file_path: str | Path, file_hash: str = "") -> tuple[str, int]:
    """任意图片/PDF 的 OCR 入口。"""
    path = Path(file_path)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        return ocr_image(path, file_hash=file_hash), 1
    if path.suffix.lower() == ".pdf":
        return ocr_pdf(path, file_hash=file_hash)
    raise ValueError(f"OCR 不支持的文件类型: {path.suffix}")
