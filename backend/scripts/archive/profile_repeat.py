"""连续多次检索计时：区分"首次冷启动"与"每次都慢"。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

Q = "消防安全重点单位多久组织演练一次？"


def main() -> int:
    print("预热 embedding 模型...")
    from app.dependencies import embed_model_dep

    embed = embed_model_dep()
    qv = embed.get_text_embedding(Q)

    from llama_index.core.vector_stores.types import VectorStoreQuery

    from app.db.vector_store import get_vector_store

    store = get_vector_store()
    print("\n=== 连续 4 次 store.query()（同一 store 实例）===")
    for i in range(4):
        t = time.perf_counter()
        store.query(VectorStoreQuery(query_embedding=qv, similarity_top_k=settings.retrieve_top_k))
        print(f"  第 {i + 1} 次: {time.perf_counter() - t:7.2f}s")

    print("\n=== 连续 3 次完整混合检索（向量+BM25+融合+精排）===")
    from app.core.retrieval.hybrid import build_hybrid_retriever
    from app.db.vector_store import build_vector_index

    index = build_vector_index()
    for i in range(3):
        t = time.perf_counter()
        r = build_hybrid_retriever(index)     # 路由里每个请求都会重新构建
        nodes = r.retrieve(Q)
        print(f"  第 {i + 1} 次: {time.perf_counter() - t:7.2f}s  召回 {len(nodes)} 条")

    print("\n=== 复用同一个 retriever 连续检索 3 次 ===")
    r = build_hybrid_retriever(index)
    for i in range(3):
        t = time.perf_counter()
        r.retrieve(Q)
        print(f"  第 {i + 1} 次: {time.perf_counter() - t:7.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
