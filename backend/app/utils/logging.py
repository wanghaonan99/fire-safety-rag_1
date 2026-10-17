"""日志配置：统一格式，本地彩色/结构化都很容易接。

用法：
    from app.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("...")

说明：lib 层（transformers / pdfplumber / httpx）的日志噪音太大，
默认压到 WARNING，需要排查时把 LOG_LEVEL=DEBUG 打开。
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "transformers",
    "sentence_transformers",
    "pdfminer",
    "PIL",
    "matplotlib",
    "ragas",
    "langchain",
    "openai",
    "asyncio",
]

_CONFIGURED = False


def setup_logging(level: str = "INFO", force: bool = False) -> None:
    """进程级日志初始化（幂等）。"""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """取 logger；顺手做一次兜底初始化（脚本直接 import 也不至于没输出）。"""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name or "fire_safety_rag")


class Timer:
    """轻量计时上下文管理器，用于打印耗时。

    with Timer("检索") as t:
        ...
    print(t.elapsed_ms)
    """

    def __init__(self, label: str = "", logger: logging.Logger | None = None, level: int = logging.INFO):
        self.label = label
        self.logger = logger
        self.level = level
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.logger:
            self.logger.log(self.level, "%s 耗时 %.1fms", self.label, self.elapsed_ms)
