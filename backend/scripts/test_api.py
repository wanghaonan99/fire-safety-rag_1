"""API 冒烟测试：用 TestClient 在进程内跑完整 FastAPI 路由，不依赖已启动的服务。

为什么要有它：
1. 一条命令验证 8 项接口是否正常，不用手工在 Swagger 里点；
2. 能复现"只有走 ASGI 才暴露"的问题——最典型的是**同步处理器写成了 async def**，
   导致 `Detected nested async` 这种在命令行脚本里永远重现不了的错误；
3. 覆盖会话记忆、语义缓存、管理接口等跨请求行为。

用法：
    cd backend
    python scripts/test_api.py                       # 用默认问题
    python scripts/test_api.py "你们厂灭火器多久检查一次？"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

DEFAULT_QUESTION = "消防安全重点单位多久组织演练一次？"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  —  {detail}" if detail else ""))
    return ok


def main() -> int:
    from fastapi.testclient import TestClient

    from app.main import app

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"API 冒烟测试：{settings.app_name}（{settings.environment}）")
    print(f"测试问题：{question}\n")

    with TestClient(app) as client:
        # ---------- 1. 健康检查 ----------
        r = client.get("/health")
        body = r.json()
        check("GET /health", r.status_code == 200 and body.get("status") == "ok",
              f"database={body.get('database')} llm={body.get('llm_configured')} cache={body.get('cache_backend')}")

        # ---------- 2. 知识库是否有资料 ----------
        r = client.get("/api/ingest/documents")
        docs = (r.json().get("data") or {}).get("items", [])
        total_chunks = sum(d.get("chunk_count", 0) for d in docs)
        check("GET /api/ingest/documents", r.status_code == 200 and len(docs) > 0,
              f"{len(docs)} 份文档 / {total_chunks} 块")
        for d in docs[:5]:
            print(f"      · {d['title'][:40]}  status={d['status']}  chunks={d['chunk_count']}")

        if not docs:
            print("\n知识库为空，先跑：python scripts/ingest_cli.py --file <文件> --category law")
            return 1

        # ---------- 3. 首次问答（含模型加载）----------
        t0 = time.perf_counter()
        r = client.post("/api/query", json={"question": question, "top_k": 5}, timeout=600)
        dt = time.perf_counter() - t0
        if not check("POST /api/query（首次）", r.status_code == 200, f"HTTP {r.status_code}，{dt:.1f}s"):
            print("\n  失败详情：", r.text[:600])
            return 1

        data = r.json()["data"]
        answer, sources = data.get("answer", ""), data.get("sources", [])
        session_id = data.get("session_id")
        check("答案非空", bool(answer.strip()), f"{len(answer)} 字")
        check("带出处引用", len(sources) > 0, f"{len(sources)} 条，"
              f"最高分 {max((s.get('score', 0) for s in sources), default=0):.3f}")
        check("答案里有 [编号] 引用", "[" in answer, "防幻觉格式要求")
        print("\n  ---- 答案 ----")
        for line in answer.splitlines()[:8]:
            print(f"  {line}")
        print("  ---- 出处 ----")
        for s in sources[:3]:
            print(f"  · score={s.get('score', 0):.3f} 《{s.get('document_title')}》{s.get('heading_path', '')}")
        print()

        # ---------- 4. 语义缓存：同一问题再问一次 ----------
        t1 = time.perf_counter()
        r2 = client.post("/api/query", json={"question": question, "top_k": 5}, timeout=600)
        dt2 = time.perf_counter() - t1
        d2 = r2.json().get("data", {})
        check("重复提问命中语义缓存", d2.get("cached") is True,
              f"cached={d2.get('cached')}，耗时 {dt2:.2f}s（首次 {dt:.1f}s）")

        # ---------- 5. 会话记忆：追问一句带指代的话 ----------
        if session_id:
            r3 = client.post(
                "/api/query",
                json={"question": "那火灾高危单位呢？", "session_id": session_id, "top_k": 5},
                timeout=600,
            )
            d3 = r3.json().get("data", {})
            check("多轮追问（指代消解）", r3.status_code == 200 and bool(d3.get("answer")),
                  f"transformed={d3.get('transformed_queries')}")

            r4 = client.get(f"/api/sessions/{session_id}/history")
            history = r4.json().get("data", [])
            check("会话历史落库", len(history) >= 4, f"{len(history)} 条消息")

        # ---------- 6. 管理接口鉴权 ----------
        r5 = client.get("/api/admin/stats")
        check("管理接口无口令被拒", r5.status_code == 401, f"HTTP {r5.status_code}")

        token = settings.admin_token
        if not token.isascii():
            # HTTP 头不允许非 ASCII 字符，httpx 会在编码阶段直接抛 UnicodeEncodeError，
            # 常见于把 .env.example 里的说明文字当值填了进去
            check(
                "管理接口带口令可用",
                False,
                "ADMIN_TOKEN 含非 ASCII 字符，请在 .env 里换成："
                'python -c "import secrets;print(secrets.token_urlsafe(32))" 的输出',
            )
        else:
            r6 = client.get("/api/admin/stats", headers={"X-Admin-Token": token})
            stats = r6.json().get("data", {})
            check("管理接口带口令可用", r6.status_code == 200,
                  f"文档 {stats.get('documents')} / 切块 {stats.get('chunks')} / 会话 {stats.get('sessions')}")

            r7 = client.post("/api/admin/cache/clear", headers={"X-Admin-Token": token})
            check("清空语义缓存", r7.status_code == 200, f"删除 {r7.json().get('data', {}).get('removed')} 项")

    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{'=' * 60}\n结果：{passed}/{len(_results)} 项通过")
    failed = [n for n, ok, _ in _results if not ok]
    if failed:
        print("未通过：" + "、".join(failed))
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
