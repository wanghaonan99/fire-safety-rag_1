"""测 Neon 的纯网络延迟（TCP / TLS）与 SQL 往返分布，区分"网络慢"还是"连接池慢"。"""

from __future__ import annotations

import socket
import ssl
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.session import engine  # noqa: E402

parts = urlsplit(settings.database_url)
HOST, PORT, DB = parts.hostname, parts.port or 5432, (parts.path or "/neondb").lstrip("/")


def tcp_connect() -> float:
    t = time.perf_counter()
    with socket.create_connection((HOST, PORT), timeout=15):
        pass
    return time.perf_counter() - t


def tls_handshake() -> float:
    t = time.perf_counter()
    ctx = ssl.create_default_context()
    with socket.create_connection((HOST, PORT), timeout=15) as raw:
        with ctx.wrap_socket(raw, server_hostname=HOST):
            pass
    return time.perf_counter() - t


def main() -> int:
    print(f"目标: {HOST}:{PORT}/{DB}（{settings.embedding_provider}）\n")

    print("=== 1. 纯网络往返 ===")
    tcp = [tcp_connect() for _ in range(4)]
    print(f"  TCP 建连   : 平均 {statistics.mean(tcp):.3f}s  {[round(x, 3) for x in tcp]}")
    tls = [tls_handshake() for _ in range(3)]
    print(f"  TLS 握手   : 平均 {statistics.mean(tls):.3f}s  {[round(x, 3) for x in tls]}")
    print(f"  → 单次网络 RTT 下限约 {statistics.mean(tls) / 2:.3f}s（TLS 握手 ≈ 2 个 RTT）")

    print("\n=== 2. SQL 往返分布（池内复用，连续 8 次）===")
    times = []
    for i in range(8):
        t = time.perf_counter()
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        times.append(time.perf_counter() - t)
    print(f"  平均 {statistics.mean(times):.3f}s  中位 {statistics.median(times):.3f}s")
    print(f"  明细 {[round(x, 3) for x in times]}")

    print("\n=== 3. 单事务内多条语句（看是否每条都贵）===")
    t = time.perf_counter()
    with engine.begin() as c:
        for _ in range(5):
            c.execute(text("SELECT 1"))
    dt = time.perf_counter() - t
    print(f"  5 条 SELECT 在 1 个事务里: {dt:.3f}s（每条 {dt / 5:.3f}s）")

    print("\n=== 4. 一次典型向量检索要几条语句 ===")
    from sqlalchemy import event

    counter = {"n": 0}

    @event.listens_for(engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        counter["n"] += 1

    try:
        from app.db.vector_store import get_vector_store

        store = get_vector_store()
        t = time.perf_counter()
        store.query(__import__("llama_index.core.vector_stores.types", fromlist=["VectorStoreQuery"]).VectorStoreQuery(
            query_embedding=[0.0] * settings.embedding_dim, similarity_top_k=10
        ))
        dt = time.perf_counter() - t
        print(f"  store.query() 耗时 {dt:.2f}s，期间执行了 {counter['n']} 条 SQL")
    except Exception as exc:  # noqa: BLE001
        print(f"  向量检索计时失败: {type(exc).__name__}: {str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
