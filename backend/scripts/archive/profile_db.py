"""只测 Neon 网络往返，不加载任何模型。

用来判断"每次提问几十秒"是 数据库冷启动（一次性）还是 每次往返都慢（结构性问题）。
用法：cd backend && python scripts/profile_db.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import SessionLocal, engine  # noqa: E402


def timed(label: str, fn, repeat: int = 1):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    avg = sum(times) / len(times)
    print(f"  {avg:7.3f}s  {label}" + (f"   （{repeat} 次明细：{', '.join(f'{x:.2f}' for x in times)}）" if repeat > 1 else ""))
    return avg


def main() -> int:
    print("=== 1. 冷连接（新建连接 + SELECT 1）===")
    timed("建立连接并 SELECT 1", lambda: engine.connect().close())

    print("\n=== 2. 池内连接复用（第 2~5 次）===")
    def one_select():
        with engine.connect() as c:
            c.execute(text("SELECT 1"))

    timed("SELECT 1", one_select, repeat=4)

    print("\n=== 3. 向量检索相关 SQL ===")
    timed("count(*) from data_fire_chunks", lambda: _scalar("SELECT count(*) FROM data_fire_chunks"))
    timed("全表取 text/metadata（BM25 语料用）", lambda: _scalar("SELECT count(*) FROM (SELECT node_id, text, metadata_ FROM data_fire_chunks) t"))

    print("\n=== 4. 业务表 ===")
    timed("count(*) documents", lambda: _scalar("SELECT count(*) FROM documents"))
    timed("count(*) chunks", lambda: _scalar("SELECT count(*) FROM chunks"))

    print("\n=== 5. 会话读写（每次提问都要做的）===")
    from app.core.memory.session_store import SessionStore

    def session_write():
        db = SessionLocal()
        try:
            store = SessionStore(db)
            s = store.get_or_create(None, user_id="profile")
            store.add_message(s.id, "user", "测试问题")
            store.add_message(s.id, "assistant", "测试回答")
            store.get_history(s.id)
        finally:
            db.close()

    timed("建立会话 + 写 2 条消息 + 读历史", session_write, repeat=2)

    print("\n=== 6. 逐条语句计时（找往返次数）===")
    db = SessionLocal()
    try:
        store = SessionStore(db)
        t = time.perf_counter()
        s = store.get_or_create(None, user_id="profile")
        print(f"  {time.perf_counter()-t:7.3f}s  get_or_create（1 次 INSERT + 1 次 SELECT）")
        t = time.perf_counter()
        store.add_message(s.id, "user", "q")
        print(f"  {time.perf_counter()-t:7.3f}s  add_message（INSERT + SELECT session + UPDATE + commit）")
        t = time.perf_counter()
        store.get_history(s.id)
        print(f"  {time.perf_counter()-t:7.3f}s  get_history（1 次 SELECT）")
    finally:
        db.close()
    return 0


def _scalar(sql: str):
    with engine.connect() as c:
        return c.execute(text(sql)).scalar()


if __name__ == "__main__":
    raise SystemExit(main())
