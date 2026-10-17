"""找出（并可选地删除）库里残留的"纯目次碎片"切块。

背景：低信息密度过滤原本只在**段落级**执行。当目次页与大段正文同属一个 section 时
不会被判定为目次，语义切块后又把它切成"纯点线 + 章节名"的小片段，这些片段会被检索召回，
从它们出的评估题也只能靠模型编造。

⚠️ 为什么推荐"直接删"而不是"重新入库"：
   重新入库 GB 51283 会生成**全新的 chunk id**，使已校对评估集里 98 条依赖该文档的
   ground_truth_ids 全部失效。直接删除这十几条碎片则完全不影响评估集。

用法：
    python scripts/remove_toc_fragments.py            # 干跑，只打印
    python scripts/remove_toc_fragments.py --apply    # 真正删除（同时删向量、修正 chunk_count）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.core.ingestion.chunker import is_low_value_chunk  # noqa: E402
from app.db.session import engine  # noqa: E402


def find_fragments(conn) -> list[tuple[str, str, str, str]]:
    """返回 [(chunk_id, document_id, title, content)]，只挑确实会被新过滤规则丢弃的块。"""
    rows = conn.execute(
        text(
            "SELECT k.id, k.document_id, d.title, k.content FROM chunks k "
            "JOIN documents d ON d.id = k.document_id WHERE k.content LIKE '%........%'"
        )
    ).all()
    out = []
    for cid, did, title, content in rows:
        low, reason = is_low_value_chunk(content)
        if low and reason == "目次":
            out.append((cid, did, title, content))
    return out


def main() -> int:
    apply = "--apply" in sys.argv
    with engine.begin() as conn:
        frags = find_fragments(conn)
        print(f"发现纯目次碎片 {len(frags)} 块\n")
        for cid, did, title, content in frags:
            print(f"  《{title[:24]}》{content[:80].replace(chr(10), ' ')}")

        if not frags:
            return 0
        if not apply:
            print(f"\n（干跑）加 --apply 才会真正删除这 {len(frags)} 块")
            return 0

        ids = [f[0] for f in frags]
        # 1) 删向量。⚠️ PGVectorStore 表里的关联列叫 **node_id**（不是 id；id 是它自己的主键），
        #    写成 id 会静默匹配 0 行、留下孤儿向量。
        deleted_vec = conn.execute(
            text("DELETE FROM data_fire_chunks WHERE node_id::text = ANY(:ids)"), {"ids": ids}
        ).rowcount
        # 2) 删业务表切块
        deleted_chunk = conn.execute(
            text("DELETE FROM chunks WHERE id::text = ANY(:ids)"), {"ids": ids}
        ).rowcount
        # 3) 修正各文档的 chunk_count，保证与向量表一致
        for did in {f[1] for f in frags}:
            conn.execute(
                text(
                    "UPDATE documents SET chunk_count = "
                    "(SELECT count(*) FROM chunks WHERE document_id = :d) WHERE id = :d"
                ),
                {"d": did},
            )
        print(f"\n✅ 已删除：向量 {deleted_vec} 行、切块 {deleted_chunk} 行，并已修正 chunk_count")
        print("   注意：后端进程内的 BM25 语料是缓存，需要重启后端才会生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
