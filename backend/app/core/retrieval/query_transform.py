"""查询改写（第二批）：HyDE + Step-back Prompting。

两种技术的分工：
- **HyDE**（Hypothetical Document Embeddings）：先让 LLM 编一段"假答案"，
  用假答案去检索。适合用户问得很口语、与法规书面语差距大的场景；
- **Step-back**：先问一个更上位的问题（"消防通道宽度要求" -> "建筑消防设计的一般原则"），
  用上位问题检索法律原则性条文，弥补具体条款缺失。

工程约束：每次改写都是一次 LLM 调用（+1~2s 延迟）。因此默认：
- ENABLE_QUERY_TRANSFORM=true 只启用 step-back（成本可控）；
- ENABLE_HYDE=false，效果不达标再开。
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

STEP_BACK_PROMPT = """你是一位消防安全领域的资深工程师。
请把用户的具体问题**抽象成更上位、更本质的一般性问题**，用于在法规原文中检索。
只输出一个问句，不要解释。

示例：
具体问题：甲类厂房与员工宿舍的防火间距应该是多少米？
上位问题：甲类厂房与人员密集场所之间的防火间距设计原则是什么？

具体问题：{question}
上位问题："""

HYDE_PROMPT = """请以《建筑设计防火规范》等中国消防法规的口吻，写一段 150 字以内的
专业文字，用来回答下面的问题。允许你不确定，但必须使用规范的术语和条款式表述。
只输出正文，不要任何前后缀。

问题：{question}
"""


def step_back_query(question: str, llm: Any = None) -> str:
    """生成上位问题；失败则原样返回。"""
    if llm is None:
        from app.dependencies import llm_dep

        llm = llm_dep()
    try:
        response = llm.complete(STEP_BACK_PROMPT.format(question=question))
        text = str(getattr(response, "text", response)).strip().splitlines()[0]
        logger.info("Step-back: %s -> %s", question, text)
        return text or question
    except Exception as exc:
        logger.warning("Step-back 失败: %s", exc)
        return question


def hyde_document(question: str, llm: Any = None) -> str:
    """生成假想答案文本（HyDE），失败则返回原问题。"""
    if llm is None:
        from app.dependencies import llm_dep

        llm = llm_dep()
    try:
        response = llm.complete(HYDE_PROMPT.format(question=question))
        return str(getattr(response, "text", response)).strip() or question
    except Exception as exc:
        logger.warning("HyDE 失败: %s", exc)
        return question


def rewrite_queries(
    question: str,
    llm: Any = None,
    enable_transform: bool | None = None,
    enable_hyde: bool | None = None,
) -> tuple[list[str], str]:
    """返回 (用于检索的查询列表, 用于生成的原始问题)。

    检索查询列表 = [原始问题] + (可选) [上位问题] + (可选) [HyDE 假想答案]，
    它们会分头检索再融合 —— 注意这会让检索次数成倍增加，按预算开。
    """
    do_transform = settings.enable_query_transform if enable_transform is None else enable_transform
    do_hyde = settings.enable_hyde if enable_hyde is None else enable_hyde

    queries = [question]
    if do_transform:
        queries.append(step_back_query(question, llm=llm))
    if do_hyde:
        queries.append(hyde_document(question, llm=llm))
    return queries, question


# ---------------------------------------------------------------- LlamaIndex 原生适配
def build_hyde_transform(llm: Any = None) -> Any:
    """使用 LlamaIndex 内置 HyDEQueryTransform（与本文件的 hyde_document 等价）。

    内置版会把生成文本 + 原问题一起返回 QueryBundle，接入 query engine 更省事。
    """
    from llama_index.core.indices.query.query_transform import HyDEQueryTransform

    if llm is None:
        from app.dependencies import llm_dep

        llm = llm_dep()
    return HyDEQueryTransform(llm=llm, include_original=True)


def build_step_back_transform(llm: Any = None) -> Any:
    """LlamaIndex StepDecomposeQueryTransform 的轻封装（用于多步推理场景）。"""
    from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform

    if llm is None:
        from app.dependencies import llm_dep

        llm = llm_dep()
    # verbose 打开可看到每步分解结果，调试期很有用
    return StepDecomposeQueryTransform(llm=llm, verbose=settings.debug)
