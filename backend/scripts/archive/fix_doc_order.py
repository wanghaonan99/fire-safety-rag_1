"""一次性修正 docs/评估方案.md 的章节顺序：把"### 3 生成基线结果"移到"### 2 检索基线结果"之后。"""

from __future__ import annotations

from pathlib import Path

DOC = Path("docs/评估方案.md")

SEC3_HEAD = "### 3 生成基线结果（RAGAS"
RETRIEVAL_TABLE = "| 指标 | 门槛 | **实测** | 结论 |"
ANCHOR_1C = "### 1c 出题结果与两个必须人工处理的问题"
SEC2_HEAD = "### 2 检索基线结果（baseline-v1，140 条，151 秒）"


def main() -> int:
    text = DOC.read_text(encoding="utf-8")

    start = text.index(SEC3_HEAD)
    table_at = text.index(RETRIEVAL_TABLE, start)
    anchor = text.index(ANCHOR_1C)

    block3 = text[start:table_at].rstrip()          # 生成基线整段
    section2 = text[table_at:anchor].rstrip()       # 检索基线整段（原属于第 2 节）

    if SEC2_HEAD in text[table_at:anchor]:
        print("已经修正过，无需处理")
        return 0

    new_text = (
        text[:start].rstrip()
        + "\n\n"
        + SEC2_HEAD
        + "\n\n"
        + section2
        + "\n\n"
        + block3
        + "\n\n"
        + text[anchor:]
    )
    DOC.write_text(new_text, encoding="utf-8")
    print("已重排：第 2 节（检索基线）→ 第 3 节（生成基线）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
