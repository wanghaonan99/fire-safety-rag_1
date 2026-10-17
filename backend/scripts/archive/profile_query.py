"""问答链路分段计时：定位"每次提问都要几十秒"到底慢在哪一段。

用法：cd backend && python scripts/profile_query.py "你的问题"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "消防安全重点单位多久组织演练一次？"
_rows: list[tuple[str, float]] = []


def stage(name: str):
    """计时上下文管理器：with stage('xxx'): ..."""

    class _T:
        def __enter__(self):
            self.t = time.perf_counter()
            return self

        def __exit__(self, *a):
            dt = time.perf_counter() - self.t
            _rows.append((name, dt))
            print(f"  {dt:7.2f}s  {name}")

    return _T()


def main() -> int:
    import torch

    print(f"问题：{QUESTION}")
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}")
    print(f"配置：rerank={settings.rerank_provider} use_async={settings.retrieval_use_async} "
          f"top_k={settings.retrieve_top_k}→{settings.rerank_top_k}\n")

    # ---------- 1. 模型加载（只在第一次慢，用于对照）----------
    with stage("① 加载 embedding 模型（仅首次）"):
        from app.dependencies import embed_model_dep

        embed = embed_model_dep()

    with stage("② 单条问题向量化（应为毫秒级）"):
        vec = embed.get_text_embedding(QUESTION)
    print(f"      向量维度 {len(vec)}")

    # ---------- 2. 向量库 ----------
    with stage("③ 连接/构建 VectorStoreIndex"):
        from app.db.vector_store import build_vector_index

        index = build_vector_index()

    with stage("④ 向量检索（含连 Neon）"):
        vr = index.as_retriever(similarity_top_k=settings.retrieve_top_k)
        vnodes = vr.retrieve(QUESTION)
    print(f"      向量召回 {len(vnodes)} 条")

    # ---------- 3. BM25 ----------
    with stage("⑤ 构建 BM25 语料（首次查全表）"):
        from app.core.retrieval.hybrid import build_bm25_retriever

        bm25 = build_bm25_retriever(top_k=settings.bm25_top_k)

    with stage("⑥ BM25 检索"):
        bnodes = bm25.retrieve(QUESTION)
    print(f"      BM25 召回 {len(bnodes)} 条")

    with stage("⑦ 再建一次 BM25（模拟每次提问都重建）"):
        bm25_again = build_bm25_retriever(top_k=settings.bm25_top_k)

    # ---------- 4. 精排（重点怀疑对象）----------
    cand = vnodes + bnodes
    print(f"      精排候选 {len(cand)} 条")
    with stage("⑧ 加载 Reranker（仅首次）"):
        from app.core.retrieval.reranker import build_reranker

        reranker = build_reranker()

    if reranker is not None:
        with stage(f"⑨ 精排 {len(cand)} 条（GPU 应 <2s）"):
            from llama_index.core.schema import QueryBundle

            reranked = reranker.postprocess_nodes(cand, query_bundle=QueryBundle(query_str=QUESTION))
        print(f"      精排后 {len(reranked)} 条")
    else:
        reranked = cand[: settings.rerank_top_k]

    # ---------- 5. LLM ----------
    with stage("⑩ Step-back 查询改写（1 次 DeepSeek）"):
        from app.core.retrieval.query_transform import step_back_query

        step_back_query(QUESTION)

    with stage("⑪ 生成答案（1 次 DeepSeek）"):
        from app.core.generation.llm import chat_direct
        from app.core.generation.prompt import build_qa_messages

        messages = build_qa_messages(QUESTION, reranked[: settings.rerank_top_k])
        prompt_chars = sum(len(m["content"]) for m in messages)
        chat_direct(messages)
    print(f"      提示词总长度 {prompt_chars} 字符")

    # ---------- 6. 数据库写入 ----------
    with stage("⑫ 会话读写（Neon 往返）"):
        from app.core.memory.session_store import SessionStore
        from app.db.session import session_scope

        with session_scope() as db:
            store = SessionStore(db)
            s = store.get_or_create(None, user_id="profile")
            store.add_message(s.id, "user", QUESTION)
            store.add_message(s.id, "assistant", "（profiling）")
            store.get_history(s.id)

    # ---------- 汇总 ----------
    print("\n" + "=" * 56)
    one_time = {"① 加载 embedding 模型（仅首次）", "⑧ 加载 Reranker（仅首次）", "⑤ 构建 BM25 语料（首次查全表）"}
    print(f"{'阶段':<40}{'耗时':>10}")
    print("-" * 56)
    total = 0.0
    steady = 0.0
    for name, dt in _rows:
        print(f"{name:<40}{dt:>9.2f}s")
        total += dt
        if name not in one_time:
            steady += dt
    print("-" * 56)
    print(f"{'合计（含一次性加载）':<40}{total:>9.2f}s")
    print(f"{'稳态（每次提问都应承担的）':<40}{steady:>9.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
