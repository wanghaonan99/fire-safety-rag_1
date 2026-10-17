"""FastAPI 依赖注入集中处。

约定：**所有重量级对象（LLM / Embedding / Reranker / VectorStore）都懒加载 + 单例**，
避免 import 时就下载模型导致启动失败或 Render 冷启动超时。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings, settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def settings_dep() -> Settings:
    """路由里 `s: Settings = Depends(settings_dep)`。"""
    return get_settings()


@lru_cache(maxsize=1)
def _embed_model_singleton() -> Any:
    """单例 Embedding 模型（bge-m3 本地加载约 2GB 内存，只能有一个实例）。"""
    from app.core.ingestion.embedder import build_embed_model

    logger.info("首次加载 Embedding 模型 ...")
    return build_embed_model()


def embed_model_dep() -> Any:
    return _embed_model_singleton()


@lru_cache(maxsize=1)
def _llm_singleton() -> Any:
    from app.core.generation.llm import build_llm

    return build_llm()


def llm_dep() -> Any:
    return _llm_singleton()


@lru_cache(maxsize=1)
def _reranker_singleton() -> Any:
    from app.core.retrieval.reranker import build_reranker

    return build_reranker()


def reranker_dep() -> Any:
    return _reranker_singleton()


_SETTINGS_CONFIGURED = False


def ensure_llama_settings() -> None:
    """把 DeepSeek LLM 与本地 embedding 注册为 LlamaIndex 的全局默认（幂等）。

    为什么必须做：LlamaIndex 内部很多组件在没显式传 llm 时会回落到 `Settings.llm`，
    而 `Settings.llm` 的默认值是 **OpenAI**，于是抛出
    `ValueError: No API key found for OpenAI`——本项目根本不用 OpenAI。
    注册之后，任何内部回落都会走 DeepSeek。

    刻意不在应用启动时调用：它会立刻加载 embedding 模型（约 2GB），
    在 Render 免费套餐上会导致启动即 OOM；交给首次真正用到时再加载。
    """
    global _SETTINGS_CONFIGURED
    if _SETTINGS_CONFIGURED:
        return
    from llama_index.core import Settings

    Settings.llm = _llm_singleton()
    Settings.embed_model = _embed_model_singleton()
    _SETTINGS_CONFIGURED = True
    logger.info("LlamaIndex 全局默认已注册：llm=%s, embed=%s",
                settings.deepseek_model, settings.embedding_model)


def reset_model_caches() -> None:
    """测试 / 换模型时清空单例。"""
    global _SETTINGS_CONFIGURED
    _embed_model_singleton.cache_clear()
    _llm_singleton.cache_clear()
    _reranker_singleton.cache_clear()
    _SETTINGS_CONFIGURED = False
