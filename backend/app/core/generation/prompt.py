"""Prompt 模板：消防问答的"有据可依 + 拒绝幻觉"约束。

三条硬规则（消防场景安全相关，不可妥协）：
1. 只依据给定资料回答，资料没写就明确说"资料中未找到"，**不许编造**；
2. 必须标注引用编号 [1][2]，方便用户核对原文；
3. 涉及强制条文/数值的，给出规范名称与条款号（有则给）。
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """你是"工厂消防知识问答助手"，服务对象是工厂安全员与一线员工。
你的回答必须严格遵守：
1. **只使用【参考资料】中的信息**作答，不得引入外部知识或凭经验推测；
2. 若参考资料不足以回答，以"根据现有资料无法确定"开头，说明资料中缺什么、
   并建议查阅**具体的那份规范**（或相关规范原文）或咨询安全管理部门；
   **禁止输出"XXX"这类占位符，也不要编造规范名称**；
3. 引用具体内容时在句末标注编号，如 [1][2]，编号对应参考资料序号；
4. 涉及数值/限值时，原文怎么写就怎么引用，不要换算或近似；
5. 用简体中文回答，按"**结论 / 依据 / 怎么做**"三段输出：结论给答案（关键数值加粗），
   依据逐条列引文并标 [编号]，怎么做只在涉及操作/判定时给（纯定义题省略）；
6. 若问题涉及紧急危险情况，先给安全提示，再给依据。"""

CONTEXT_TEMPLATE = """【参考资料】
{context}
【参考资料结束】"""

QA_PROMPT_TEMPLATE = """{system}

{context}

用户问题：{question}

请**严格按下面的三段结构**输出（这是一份给安全员看的答复，不是聊天）：

- 结论：
  直接回答问题，**1~2 句**，不要罗列补充信息。
  关键数值与关键判断词用 **加粗**（如"不应小于 **30m**"），便于快速扫读。
  事实性陈述在句末标 [编号]。

- 依据：
  **必须逐条列出**，每条单独一行，统一格式：
  - 《文档名》条款号/章节：原文引述（用引号括起）[编号]
  例：- 《精细化工企业工程设计防火标准（GB 51283-2020）》4.1.5：原文"精细化工企业与相邻工厂或设施的防火间距不应小于表4.1.5的规定。"[1]
  **不要**把依据写成一大段散文，也不要合并多条为一条。

- 怎么做：
  只在问题涉及**具体判定、操作、流程**时输出，用有序列表写"资料明确要求"的动作，每条都标 [编号]。
  **纯定义或纯事实类问题（如"XX 是指什么"）请整段省略**，不要为了凑格式而编造动作。
"""

CONDENSE_PROMPT_TEMPLATE = """以下是历史对话与用户的新问题。请把新问题改写成一个**不依赖上下文、
可独立检索**的完整问题**（只输出改写后的问题，不要回答它）**。

历史对话：
{history}

用户新问题：{question}
独立问题："""


TABLE_CHUNK_MAX_CHARS = 3500      # 含表格的切块最多给到这么多字符


def _clip_for_prompt(text: str, limit: int) -> str:
    """按上限裁剪切块，但**含 Markdown 表格的切块要保证表格不被截掉**。

    实测问题（GB 51283 表4.1.5 防火间距表）：切块 1824 字符，表格从第 934 字符开始，
    而原来的上限是 1200 字符 —— 表格的数据行刚好被切掉，模型只能回答
    "资料中未给出完整表格数值，需查表"，明明库里有数值却答不出来。
    表格是数值型答案的唯一来源，所以给它一个更宽松但仍有限的上限。
    """
    if len(text) <= limit:
        return text
    if "【表格" in text:
        return text[: max(limit, TABLE_CHUNK_MAX_CHARS)]
    return text[:limit]


def format_context(nodes: list[Any], max_chars_per_node: int | None = None) -> str:
    """把检索节点渲染成带编号的参考资料块。"""
    if max_chars_per_node is None:
        from app.config import settings

        max_chars_per_node = settings.prompt_node_max_chars

    blocks: list[str] = []
    for i, n in enumerate(nodes, 1):
        node = getattr(n, "node", n)
        meta = node.metadata or {}
        title = meta.get("title", "未知资料")
        heading = meta.get("heading_path", "")
        page = meta.get("page_no", 0)
        loc = f"《{title}》" + (f" {heading}" if heading else "") + (f" 第{page}页" if page else "")
        blocks.append(f"[{i}] {loc}\n{_clip_for_prompt(node.get_content(), max_chars_per_node)}")
    return "\n\n".join(blocks)


def build_qa_messages(
    question: str,
    nodes: list[Any],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """组装最终发给 DeepSeek 的 messages（OpenAI 格式）。

    history 形如 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if turn.get("role") in {"user", "assistant"}:
            messages.append({"role": turn["role"], "content": turn["content"]})

    context = format_context(nodes)
    messages.append(
        {
            "role": "user",
            "content": QA_PROMPT_TEMPLATE.format(
                system="",  # system 已单独成条，这里留空避免重复
                context=CONTEXT_TEMPLATE.format(context=context or "（无检索结果）"),
                question=question,
            ),
        }
    )
    return messages


def build_condense_prompt(question: str, history: list[dict[str, str]]) -> str:
    """多轮场景：把"它/这个"等指代补全成独立问题（检索前必须先做）。"""
    history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history[-4:])
    return CONDENSE_PROMPT_TEMPLATE.format(history=history_text or "（无）", question=question)
