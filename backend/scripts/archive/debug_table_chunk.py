"""诊断：含表格的切块到底长什么样、表格在切块里的位置。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402
from app.core.generation.prompt import format_context  # noqa: E402


class _N:
    """模拟 LlamaIndex 节点，用来复用 format_context 看截断效果。"""

    def __init__(self, content: str, heading: str = "", title: str = ""):
        self._c = content
        self.metadata = {"heading_path": heading, "title": title, "page_no": 14}

    def get_content(self, metadata_mode=None) -> str:
        return self._c


def main() -> int:
    with engine.connect() as c:
        rows = c.execute(
            text(
                "SELECT k.heading_path, k.content, length(k.content) FROM chunks k "
                "WHERE k.content LIKE '%| 90 |%'"
            )
        ).all()

    print(f"含防火间距表数值的切块：{len(rows)} 个\n")
    for heading, content, length in rows:
        has_table = "【表格" in content
        pos = content.find("【表格")
        print("=" * 80)
        print(f"标题: {heading[:50]}")
        print(f"长度: {length} 字符 | 含【表格】标记: {has_table} | 表格起始位置: {pos}")
        if pos >= 0:
            print(f"表格在切块中的位置占比: {pos / max(length, 1):.0%}")
            preview = content[pos : pos + 120].replace("\n", " ")
            print(f"表格开头: {preview}")
        print(f"切块开头: {content[:100].replace(chr(10), ' ')}")

        # 复用真实的提示词格式化函数，看截断后表格还在不在
        ctx = format_context([_N(content, heading, "GB 51283")])
        print(f"格式化后长度: {len(ctx)} | 截断后含表格数值 90: {'| 90 |' in ctx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
