"""测试公共 fixture。

原则：单元测试**不连真实 PostgreSQL、不下载模型**。
- 数据库相关：用 SQLite 内存库 + 打桩（或 pytest.mark.integration 跳过）；
- 模型相关：注入假 embed_model / 假 llm。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# 让 `from app.xxx import ...` 在 backend/ 下可直接跑
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("ENABLE_HYPE", "false")
os.environ.setdefault("REDIS_URL", "")


class FakeLLM:
    """假 LLM：complete() 直接返回预设文本，用于 HyPE / 改写 / 生成的单测。"""

    def __init__(self, text: str = '["问题一","问题二","问题三"]'):
        self.text = text
        self.calls: list[str] = []

    def complete(self, prompt: str, **_: Any):
        self.calls.append(prompt)

        class _Resp:
            def __init__(self, t: str):
                self.text = t

        return _Resp(self.text)


class FakeEmbedding:
    """假 Embedding：用字符哈希造一个固定维度向量，避免加载 2GB 模型。"""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        return [((hash(text[i % max(len(text), 1)]) % 100) / 100.0) for i in range(self.dim)]

    def get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    def get_text_embedding_batch(self, texts: list[str], **_: Any) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def get_query_embedding(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def work_tmp():
    """工作区内的临时目录（不用系统 temp：某些受限运行环境不允许写系统临时目录）。"""
    import shutil

    path = ROOT / ".pytest_work"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embed() -> FakeEmbedding:
    return FakeEmbedding()
