"""切块：语义切块 + 标题链拼接。

为什么不用固定窗口切块？
消防法规的条款（"第 5.3.2 条 ..."）语义边界明确，固定窗口会把条款截断，
导致检索到的片段答非所问。SemanticSplitterNodeParser 用 embedding 相似度找断点。

兜底：语义切块需要 embedding（慢、吃内存），可用 CHUNK_STRATEGY=token 切回
SentenceSplitter / TokenTextSplitter，仅改配置不改代码。
"""

from __future__ import annotations

import os
import re

from app.config import settings
from app.models.document import ParsedDocument
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 中文标题识别：Markdown 标题 / 第X章 / 第X条 / 数字编号
#
# ⚠️ 数字编号后面的空格必须是**可选**的（\s*）：归一化会去掉"编号与汉字之间的空格"
#    （"6.8.5.1 疏散引导行动" → "6.8.5.1疏散引导行动"），若这里强制要求 \s+，
#    标题会全部识别不出来，段落被合并、切块数量骤减（实测 64 块 → 42 块）。
HEADING_PATTERNS = [
    re.compile(r"^(#{1,6})\s+(.+)$"),                                  # Markdown 标题
    re.compile(r"^第\s*[一二三四五六七八九十百零\d]+\s*[章节条款]\s*(.*)$"),  # 第3章 / 第 4 条
    re.compile(r"^(\d+(?:\.\d+){1,4})\s*(\S.*)$"),                     # 3.2.1 xxx（空格可省）
    re.compile(r"^[（(][一二三四五六七八九十\d]+[)）]\s*(\S.*)$"),        # （三）xxx
]

# 语义切块参数：buffer_size 是"多大的窗口内比较相似度"，太大反而丢语义边界
SEMANTIC_BUFFER_SIZE = 1
SEMANTIC_BREAKPOINT_PERCENTILE = 95


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 60:
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)


def split_by_heading(text: str) -> list[tuple[str, str]]:
    """按标题把全文切成 [(heading_path, section_text)]。

    heading_path 形如 "第三章 消防设施 > 3.2 灭火器配置"，
    切块后会拼到每个 chunk 的正文前，保证片段自带上下文（关键！）。
    TODO(第二批): 支持多级标题栈 + 表格区域保护，避免条款被表格打断。
    """
    sections: list[tuple[str, str]] = []
    current_path = ""
    buffer: list[str] = []

    for line in text.splitlines():
        if _is_heading(line):
            if buffer:
                sections.append((current_path, "\n".join(buffer).strip()))
                buffer = []
            current_path = line.lstrip("#").strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_path, "\n".join(buffer).strip()))

    return [(path, body) for path, body in sections if body]


def build_node_parser(embed_model=None):
    """构造节点解析器（语义优先，失败降级 token 切块）。"""
    strategy = os.getenv("CHUNK_STRATEGY", "semantic").lower()

    if strategy == "token" or embed_model is None:
        from llama_index.core.node_parser import SentenceSplitter

        logger.info("使用 SentenceSplitter 切块")
        return SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser

        logger.info("使用 SemanticSplitterNodeParser 语义切块")
        return SemanticSplitterNodeParser(
            buffer_size=SEMANTIC_BUFFER_SIZE,
            breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_PERCENTILE,
            embed_model=embed_model,
            # 语义切块后再按 token 上限做一次硬切，防止超长节点
            include_metadata=True,
        )
    except Exception as exc:
        logger.warning("语义切块器不可用，降级 SentenceSplitter: %s", exc)
        from llama_index.core.node_parser import SentenceSplitter

        return SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


def _is_toc(text: str, heading: str) -> bool:
    """目次 / Contents。

    判据是**点线字符的密度**，而不是"有多少行含点线"：
    pdfplumber 常把整页目次合并成**一行**（"10.5防排烟 ....... 11电气 ......."），
    按行统计会漏判（实测漏掉 17 块纯目次碎片，它们被检索召回、还会孵化出幻觉题）。
    正常条文里不会出现 4 个以上连续英文句点，所以点线很长的块一定是目次。
    """
    if any(k in heading for k in ("目 次", "目次", "Contents", "CONTENTS")):
        return True

    dot_runs = re.findall(r"\.{4,}", text)
    if not dot_runs:
        return False

    dot_chars = sum(len(run) for run in dot_runs)
    compact_len = len(re.sub(r"\s", "", text)) or 1
    # 必须"点线占**多数**"才算目次。放宽到 0.25 会误杀长条文：实测有一块含
    # 11.5.5 手动报警按钮条款 + 附录A 起止点定义（点线占了后 900 字符、占比 0.40），
    # 它是有价值内容，不能删。
    return dot_chars >= 20 and dot_chars / compact_len > 0.5


def _is_cover(text: str, heading: str) -> bool:
    """封面：**没有完整句子**的版式文字（发布/实施/国家标准）。

    ⚠️ 早期版本只判断"短 + 含发布/国家标准 + 没有 应/不/宜"，结果把《消防法》第三十六条
    的正文误杀了（该段被多栏交错切开后恰好不含"应"字）。所以这里加上"不得含句末标点"
    这一条硬约束——封面不会有句号，正文一定会有。
    """
    t = text.strip()
    if len(t) > 100:
        return False
    if re.search(r"[。；！？]", t):        # 有句末标点 → 是正文，不是封面
        return False
    markers = sum(1 for k in ("发布", "实施", "联合发布", "国家标准", "中华人民共和国") if k in t)
    return markers >= 2


def _is_preface(text: str, heading: str) -> bool:
    """前言 / 起草人名单：属于版式信息，不承载可问答的知识。"""
    if any(k in heading for k in ("前 言", "前言", "起草")):
        return True
    return any(k in text for k in ("主要起草人", "起草单位", "本标准由", "本规范由"))


def _is_reference_list(text: str, heading: str) -> bool:
    """参考文献：标题为"参考文献"，或大多数行以 [n] 开头。"""
    if "参考文献" in heading or "参考文献" in text[:40]:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    ref_lines = sum(1 for ln in lines if re.match(r"^\s*\[\d+\]", ln))
    return ref_lines / len(lines) > 0.5


# 按顺序检查；命中即认为该段落不值得入库
LOW_VALUE_RULES = (
    ("目次", _is_toc),
    ("封面", _is_cover),
    ("前言/起草信息", _is_preface),
    ("参考文献", _is_reference_list),
)


def is_low_value_chunk(text: str, heading_path: str = "") -> tuple[bool, str]:
    """判断一个段落是否为"低信息密度内容"，返回 (是否丢弃, 原因)。

    这类内容（封面、目次、前言、起草人名单、参考文献）有两个危害：
      1. 它们会被向量检索召回，占掉 top-k 名额（实测发生过）；
      2. 从它们出的评估题，答案只能靠模型编造（阶段 1 里 13 条"模板占位句"废题就是同类问题）。
    """
    if not text.strip():
        return True, "空段落"
    for reason, rule in LOW_VALUE_RULES:
        try:
            if rule(text, heading_path):
                return True, reason
        except Exception:  # noqa: BLE001  单条规则异常不影响整体
            continue
    return False, ""


def chunk_document(
    parsed: ParsedDocument,
    document_id: str,
    embed_model=None,
    doc_category: str = "general",
) -> list[dict]:
    """把 ParsedDocument 切成 chunk 字典列表（不依赖 LlamaIndex Document，便于单测）。

    返回：[{content, heading_path, chunk_index, token_count, metadata}]
    """
    from llama_index.core import Document as LIDocument

    parser = build_node_parser(embed_model)
    sections = split_by_heading(parsed.text) or [("", parsed.text)]

    # 低信息密度段落（目次/封面/前言/参考文献）先丢掉：它们会被检索召回成噪声，
    # 实测在 top-5 里出现过封面块与参考文献块，白白占掉召回名额。
    if settings.enable_chunk_filter:
        kept: list[tuple[str, str]] = []
        dropped: list[str] = []
        for heading_path, body in sections:
            low, reason = is_low_value_chunk(body, heading_path)
            if low:
                dropped.append(f"{reason}({len(body)}字)")
            else:
                kept.append((heading_path, body))
        if kept:
            if dropped:
                logger.info("过滤掉 %s 个低信息密度段落：%s", len(dropped), "、".join(dropped))
            sections = kept
        else:
            logger.warning("低信息密度过滤把所有段落都过滤掉了，回退为不过滤（请检查规则）")

    chunks: list[dict] = []
    idx = 0

    for heading_path, body in sections:
        if not body.strip():
            continue
        # 关键：把标题链拼进正文，让 embedding 也"看到"章节语义
        prefixed = f"{heading_path}\n{body}" if heading_path else body
        li_doc = LIDocument(
            text=prefixed,
            metadata={
                "document_id": document_id,
                "title": parsed.title,
                "heading_path": heading_path,
                "doc_category": doc_category,
                "source_type": parsed.source_type,
                "is_scanned": parsed.is_scanned,
            },
            excluded_embed_metadata_keys=["document_id", "doc_category", "source_type", "is_scanned"],
            excluded_llm_metadata_keys=["document_id", "source_type"],
        )
        nodes = parser.get_nodes_from_documents([li_doc])

        for node in nodes:
            content = node.get_content().strip()
            if len(content) < 20:      # 丢弃目录页码之类的碎片
                continue
            # ⚠️ 段落级过滤不够：目次页与大段正文同属一个 section 时不会被判定为目次，
            #    语义切块后又把它切成"纯点线+章节名"的小片段。实测这样漏掉 17 块，
            #    它们会被检索召回，从它们出的评估题也只能靠模型编（见评估方案 3.4 的踩坑）。
            if settings.enable_chunk_filter:
                low, reason = is_low_value_chunk(content)
                if low:
                    logger.debug("丢弃低信息密度切块（%s）：%s", reason, content[:40])
                    continue
            chunks.append(
                {
                    "content": content,
                    "heading_path": heading_path,
                    "chunk_index": idx,
                    "token_count": len(content) // 2,  # 中文约 2 字符 ≈ 1 token 的粗估
                    "page_no": _guess_page_no(content),
                    "metadata": dict(node.metadata),
                }
            )
            idx += 1

    logger.info("切块完成：%s -> %s 块", parsed.title, len(chunks))
    return chunks


def _guess_page_no(content: str) -> int:
    """从 loader 插入的【第 N 页】标记里还原页码，用于溯源展示。"""
    m = re.search(r"【第\s*(\d+)\s*页】", content)
    return int(m.group(1)) if m else 0
