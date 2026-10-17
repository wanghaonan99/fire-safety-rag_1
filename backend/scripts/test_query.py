"""端到端问答验证：检索 -> 提示词 -> DeepSeek 生成 -> 会话记忆落库。

用法：cd backend && python scripts/test_query.py "你的问题"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.core.generation.llm import chat_direct  # noqa: E402
from app.core.generation.prompt import build_qa_messages  # noqa: E402
from app.core.memory.session_store import SessionStore  # noqa: E402
from app.core.retrieval.hybrid import build_hybrid_retriever  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.db.vector_store import build_vector_index  # noqa: E402

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "社会单位灭火和应急疏散预案应当包括哪些内容？"


def main() -> int:
    print(f"问题：{QUESTION}")
    print(f"检索配置：rerank={settings.rerank_provider} top_k={settings.retrieve_top_k} "
          f"→ 精排 top {settings.rerank_top_k}\n")

    t0 = time.perf_counter()
    index = build_vector_index()
    retriever = build_hybrid_retriever(index)
    nodes = retriever.retrieve(QUESTION)
    print(f"检索到 {len(nodes)} 条，耗时 {time.perf_counter() - t0:.1f}s")
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        snippet = n.node.get_content()[:55].replace("\n", " ")
        print(f"  [{i}] score={getattr(n, 'score', 0) or 0:.4f} "
              f"《{md.get('title')}》{md.get('heading_path', '')} | {snippet}…")

    t1 = time.perf_counter()
    answer = chat_direct(build_qa_messages(QUESTION, nodes))
    print(f"\n生成耗时 {time.perf_counter() - t1:.1f}s\n答案：\n{answer}\n")

    # 会话记忆：写入并读回，验证 chat_sessions / chat_messages 两张表
    sources = [
        {"chunk_id": (n.node.metadata or {}).get("chunk_id", ""), "score": float(getattr(n, "score", 0) or 0)}
        for n in nodes
    ]
    with session_scope() as db:
        store = SessionStore(db)
        session = store.get_or_create(None, user_id="cli-test")
        store.add_message(session.id, "user", QUESTION)
        store.add_message(session.id, "assistant", answer, sources=sources)
        history = store.get_history(session.id)
        print(f"会话记忆 ✓ session={session.id} 历史 {len(history)} 条 引用 {len(sources)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
