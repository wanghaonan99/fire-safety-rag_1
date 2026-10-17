"""修正 docs/评估方案.md 的章节归属（第二次，用唯一锚点）。

当前错位情况：
    ### 2 检索基线结果        ← 标题下面挂的却是 RAGAS 内容
    <RAGAS 内容>
    <检索基线内容（无标题）>
    ### 3 生成基线结果        ← 空标题，后面直接是 ### 1c
    ### 1c 出题结果...

目标：
    ### 2 检索基线结果   <检索内容>
    ### 3 生成基线结果   <RAGAS 内容>
    ### 1c 出题结果...
"""

from __future__ import annotations

from pathlib import Path

DOC = Path("docs/评估方案.md")

A = "### 2 检索基线结果（baseline-v1，140 条，151 秒）"
SEC3 = "### 3 生成基线结果（RAGAS，140 条，35 分钟）"
# 唯一锚点：检索基线表格的首行数据（RAGAS 表格里没有 HitRate）
B = "| HitRate@10 |"
D = "### 1c 出题结果与两个必须人工处理的问题"


def main() -> int:
    text = DOC.read_text(encoding="utf-8")

    a_at = text.index(A)
    b_at = text.index(B)
    c_at = text.index(SEC3)
    d_at = text.index(D)

    if not (a_at < b_at < c_at < d_at):
        print(f"锚点顺序异常：A={a_at} B={b_at} C={c_at} D={d_at}")
        return 1

    # 检索表格的实际起点：B 所在表格的上一行（表头行）
    table_head = text.rindex("| 指标 | 门槛 | **实测** | 结论 |", a_at, b_at)

    rag_content = text[a_at + len(A) : table_head].rstrip()
    retrieval_content = text[table_head:c_at].rstrip()

    new_text = (
        text[:a_at].rstrip()
        + "\n\n"
        + A
        + "\n\n"
        + retrieval_content
        + "\n\n"
        + SEC3
        + "\n\n"
        + rag_content
        + "\n\n"
        + text[d_at:]
    )
    DOC.write_text(new_text, encoding="utf-8")
    print("已修正章节归属")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
