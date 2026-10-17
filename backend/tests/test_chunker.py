"""切块相关的单元测试（不依赖数据库与模型）。"""

from __future__ import annotations

from app.core.ingestion.chunker import _guess_page_no, split_by_heading
from app.core.ingestion.loader import clean_text

SAMPLE = """# 第三章 消防设施

## 3.1 灭火器配置

一个计算单元内配置的灭火器数量不得少于 2 具。

## 3.2 消火栓

室内消火栓的间距不应大于 30m。

第 4 条 消防通道

消防车道的净宽度和净空高度均不应小于 4m。
"""


def test_split_by_heading_recognizes_headings():
    sections = split_by_heading(SAMPLE)
    paths = [p for p, _ in sections]
    assert any("灭火器配置" in p for p in paths)
    assert any("消火栓" in p for p in paths)
    # 每个 section 都要有正文
    assert all(body.strip() for _, body in sections)


def test_split_by_heading_数量合理():
    sections = split_by_heading(SAMPLE)
    assert len(sections) >= 3


def test_clean_text_压缩空行():
    raw = "第一行\n\n\n\n第二行\u3000 带全角空格"
    out = clean_text(raw)
    assert "\n\n\n" not in out
    assert "\u3000" not in out


def test_guess_page_no():
    assert _guess_page_no("【第 3 页】\n3.1 灭火器") == 3
    assert _guess_page_no("没有页码") == 0
