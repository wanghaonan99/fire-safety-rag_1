"""缓存与健康检查的冒烟测试（不依赖 Redis）。"""

from __future__ import annotations

from app.utils.cache import LRUCache


def test_lru_eviction():
    cache = LRUCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")          # a 变最近使用
    cache.set("c", 3)       # 淘汰最久未用的 b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_lru_clear():
    cache = LRUCache(maxsize=4)
    cache.set("x", 1)
    cache.clear()
    assert len(cache) == 0


def test_cosine_similarity():
    from app.core.ingestion.embedder import cosine_similarity

    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([], [1]) == 0.0


def test_health_endpoint():
    """TestClient 只验证 /health 结构；数据库/Redis 探测失败也应返回 200。"""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "database" in body and "llm_configured" in body
