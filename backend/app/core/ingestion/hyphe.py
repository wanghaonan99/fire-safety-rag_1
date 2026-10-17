"""HyPE（Hypothetical Prompt Embeddings，第二批）：为每个 chunk 生成假想问题。

原理：用户提问的口语化表达与法规原文的书面表达差距很大，
直接把 chunk 拿去 embedding 召回率低。做法是让 LLM 为每个 chunk 生成 3 个
"这段文字能回答的问题"，把这些**问题也一起索引**，检索时更容易与用户措辞对齐。

落到本项目：假想问题写进节点 metadata 的 `hypothetical_questions`，
并在建索引时把问题文本拼到被 embedding 的内容里（见 build_hype_text）。
"""

from __future__ import annotations

import json
import re

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

HYPE_QUESTIONS = 3

HYPE_PROMPT = """你是工厂消防安全知识库的语料加工助手。
请阅读下面的资料片段，生成 {n} 个**工厂员工/安全员最可能提问的问题**，
这些问题必须能由该片段直接回答。要求：
1. 用口语化中文，像一线员工在问；
2. 覆盖不同角度（判定标准、操作步骤、数值限值、责任分工等）；
3. 只输出 JSON 数组，不要任何解释，例如：["问题1","问题2","问题3"]。

资料片段：
---
{content}
---
"""


def _parse_questions(raw: str, n: int) -> list[str]:
    """LLM 输出解析：优先 JSON，失败则按行拆分兜底。"""
    text = raw.strip()
    # 去掉 ```json ``` 包裹
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(q).strip() for q in data if str(q).strip()][:n]
    except json.JSONDecodeError:
        pass
    lines = [re.sub(r"^\s*[\d\-\*\.\)、]+\s*", "", ln).strip(' ",') for ln in text.splitlines()]
    return [ln for ln in lines if len(ln) > 4][:n]


def generate_questions(content: str, llm=None, n: int = HYPE_QUESTIONS) -> list[str]:
    """为单个 chunk 生成 n 个假想问题；任何异常都返回空列表（不阻断导入）。"""
    if not settings.enable_hype or llm is None:
        return []
    try:
        prompt = HYPE_PROMPT.format(n=n, content=content[:2000])   # 截断控制成本
        if hasattr(llm, "complete"):
            response = llm.complete(prompt)
            raw = str(getattr(response, "text", response))
        else:  # 传入的是原生函数/客户端时的兜底
            raw = str(llm(prompt))
        questions = _parse_questions(raw, n)
        return questions
    except Exception as exc:
        logger.warning("HyPE 生成失败（跳过）: %s", exc)
        return []


def attach_hype(chunks: list[dict], llm=None, n: int = HYPE_QUESTIONS) -> list[dict]:
    """批量给 chunk 挂 hypothetical_questions。

    TODO(第二批优化):
      - 并发调用（`asyncio.gather` 或线程池）把导入耗时压到 1/5；
      - 结果落库到 chunks.hypothetical_questions，重建索引时可直接复用不再调 LLM；
      - 低成本替代方案：改用"关键词/实体抽取"生成伪问题，零 LLM 成本。
    """
    if not settings.enable_hype or llm is None:
        logger.info("HyPE 未启用，跳过假想问题生成")
        return chunks

    for i, chunk in enumerate(chunks):
        chunk.setdefault("metadata", {})
        chunk["metadata"]["hypothetical_questions"] = generate_questions(chunk["content"], llm=llm, n=n)
        if (i + 1) % 10 == 0:
            logger.info("HyPE 进度 %s/%s", i + 1, len(chunks))
    return chunks


def build_hype_text(chunk: dict) -> str:
    """把假想问题拼进被 embedding / 被 BM25 索引的文本。

    注意：chunk["content"] 仍是原文，用于展示与喂给 LLM；
    只有索引文本被增强（避免污染答案与引用）。
    """
    questions = (chunk.get("metadata") or {}).get("hypothetical_questions") or []
    if not questions:
        return chunk["content"]
    q_block = "\n".join(f"Q: {q}" for q in questions)
    return f"{q_block}\n\n{chunk['content']}"
