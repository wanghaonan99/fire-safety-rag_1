"""DeepSeek LLM 封装（llama-index-llms-openai-like）。

DeepSeek 完全兼容 OpenAI 协议，所以用 OpenAILike 而非自写 HTTP 客户端：
好处是自动获得 LlamaIndex 的流式、回调（LangFuse）、token 统计、重试等能力。

同时提供一个 `chat_direct()`，用于不经过 LlamaIndex 的裸调用
（RAGAS 评估、HyPE 批处理等对并发控制要求高的场景）。
"""

from __future__ import annotations

from typing import Any, Iterator

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# DeepSeek 上下文窗口 64k，输出上限 8k
DEEPSEEK_CONTEXT_WINDOW = 64000
DEEPSEEK_MAX_OUTPUT = 8192


def _v1_base() -> str:
    """OpenAILike 需要带 /v1 的 base_url；DeepSeek 两种写法都支持，这里统一补全。"""
    base = settings.deepseek_base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def build_llm(**overrides: Any) -> Any:
    """构造 DeepSeek LLM 实例（LlamaIndex 接口）。"""
    from llama_index.llms.openai_like import OpenAILike

    if not settings.deepseek_api_key:
        # 不抛异常：允许先跑通导入/检索链路，问答时再提示
        logger.warning("DEEPSEEK_API_KEY 未配置，问答功能将不可用")

    params: dict[str, Any] = {
        "model": settings.deepseek_model,
        "api_key": settings.deepseek_api_key,
        "api_base": _v1_base(),
        "temperature": settings.deepseek_temperature,
        "max_tokens": settings.deepseek_max_tokens,
        "context_window": DEEPSEEK_CONTEXT_WINDOW,
        "is_chat_model": True,
        "is_function_calling_model": False,   # deepseek-chat 暂不用 function call
        "timeout": 60.0,
        "max_retries": 3,
    }
    params.update(overrides)
    logger.info("构建 DeepSeek LLM: %s @ %s", params["model"], params["api_base"])
    return OpenAILike(**params)


def chat_direct(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """裸 HTTP 调用（同步）。失败抛异常，由调用方决定降级策略。"""
    import httpx

    payload = {
        "model": settings.deepseek_model,
        "messages": messages,
        "temperature": settings.deepseek_temperature if temperature is None else temperature,
        "max_tokens": settings.deepseek_max_tokens if max_tokens is None else max_tokens,
    }
    headers = {"Authorization": f"Bearer {settings.deepseek_api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{settings.deepseek_base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def stream_chat(messages: list[dict[str, str]]) -> Iterator[str]:
    """流式调用（供 SSE 接口使用）。

    TODO(第二批): routes_query 里接 StreamingResponse，把 token 逐个推给前端。
    """
    import httpx

    payload = {
        "model": settings.deepseek_model,
        "messages": messages,
        "temperature": settings.deepseek_temperature,
        "max_tokens": settings.deepseek_max_tokens,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {settings.deepseek_api_key}", "Content-Type": "application/json"}
    with httpx.stream(
        "POST", f"{settings.deepseek_base_url}/chat/completions", json=payload, headers=headers, timeout=120.0
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            import json

            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
