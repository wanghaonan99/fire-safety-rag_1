"""语义缓存（第二批）：Redis + 余弦相似度。

与普通 KV 缓存的区别：
用户问"灭火器多久检查一次"和"灭火器检查周期是多少"字面不同、语义相同，
普通缓存命中不了。语义缓存把问题向量化，与已缓存问题比相似度，
超过阈值（默认 0.95）就直接返回上次的答案。

三级降级策略（保证任何环境下都能跑）：
1. Redis 可用 -> Redis 存向量 + 答案，带 TTL；
2. Redis 不可用 -> 进程内 LRU（第一批默认形态，单机有效）；
3. 都不行 -> 直接透传（不缓存）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

REDIS_KEY_PREFIX = "fsrag:semcache:"
REDIS_INDEX_KEY = "fsrag:semcache:index"     # 记录所有缓存条目的 key，便于遍历比对


class LRUCache:
    """线程安全的最小 LRU（不依赖 cachetools）。"""

    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return list(self._data.items())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


_MEMORY_CACHE = LRUCache(maxsize=512)
_REDIS_CLIENT: Any = None
_REDIS_FAILED = False


def get_redis() -> Any | None:
    """懒连接 Redis；连不上就永久降级（避免每次请求都超时等待）。"""
    global _REDIS_CLIENT, _REDIS_FAILED
    if _REDIS_FAILED or not settings.redis_url:
        return None
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    try:
        import redis

        client = redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=1.5)
        client.ping()
        _REDIS_CLIENT = client
        logger.info("Redis 已连接")
    except Exception as exc:
        logger.warning("Redis 不可用，语义缓存降级为内存 LRU: %s", exc)
        _REDIS_FAILED = True
        _REDIS_CLIENT = None
    return _REDIS_CLIENT


def cache_available() -> bool:
    """健康检查用：Redis 是否真的连上了（未连通即已降级为内存 LRU）。"""
    return get_redis() is not None


def cache_backend() -> str:
    """当前生效的缓存后端：redis | memory | off。"""
    if not settings.semantic_cache_enabled:
        return "off"
    return "redis" if get_redis() is not None else "memory"


def _fingerprint(question: str, doc_category: str | None, top_k: int) -> str:
    """缓存条目的稳定 key（检索参数不同不能复用缓存）。"""
    raw = f"{question.strip()}|{doc_category or ''}|{top_k}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _embed(text: str) -> list[float]:
    """语义缓存专用的向量化。

    ⚠️ 必须用**单例** embed_model_dep()，不能用 build_embed_model()：
    后者每次调用都会新建一个 HuggingFaceEmbedding 并**重新加载 2.2GB 的 bge-m3**
    （实测每次 15~25 秒，还会反复占用一份显存）。
    语义缓存在每次问答里会被调用两次（查缓存 + 写缓存），
    用错的话每次提问凭空多花几十秒——这是本项目出现过的最贵的性能 bug。
    """
    try:
        from app.dependencies import embed_model_dep

        return embed_model_dep().get_text_embedding(text)
    except Exception as exc:
        logger.debug("缓存向量化失败: %s", exc)
        return []


def get_cached(
    question: str,
    doc_category: str | None = None,
    top_k: int = 5,
    threshold: float | None = None,
) -> dict[str, Any] | None:
    """查语义缓存：命中返回 {"answer","sources","similarity","key"}，否则 None。"""
    if not settings.semantic_cache_enabled:
        return None

    threshold = settings.semantic_cache_threshold if threshold is None else threshold
    query_vec = _embed(question)
    if not query_vec:
        return None

    from app.core.ingestion.embedder import cosine_similarity

    client = get_redis()
    entries: list[tuple[str, dict]] = []

    if client is not None:
        try:
            keys = client.zrange(REDIS_INDEX_KEY, 0, -1)
            for key in keys:
                payload = client.get(key)
                if payload:
                    entries.append((key, json.loads(payload)))
        except Exception as exc:
            logger.warning("读取 Redis 缓存失败: %s", exc)
            entries = []
    else:
        entries = _MEMORY_CACHE.items()

    best: tuple[float, str, dict] | None = None
    now = time.time()
    for key, entry in entries:
        if entry.get("expire_at") and entry["expire_at"] < now:
            continue
        if entry.get("scope") != f"{doc_category or ''}|{top_k}":
            continue   # 检索范围不同，不可复用
        sim = cosine_similarity(query_vec, entry.get("embedding", []))
        if best is None or sim > best[0]:
            best = (sim, key, entry)

    if best and best[0] >= threshold:
        sim, key, entry = best
        logger.info("语义缓存命中：similarity=%.4f", sim)
        return {"answer": entry["answer"], "sources": entry.get("sources", []), "similarity": sim, "key": key}
    return None


def set_cached(
    question: str,
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    doc_category: str | None = None,
    top_k: int = 5,
) -> bool:
    """写入语义缓存。"""
    if not settings.semantic_cache_enabled or not answer:
        return False

    vec = _embed(question)
    if not vec:
        return False

    key = REDIS_KEY_PREFIX + _fingerprint(question, doc_category, top_k)
    payload = {
        "question": question,
        "embedding": vec,
        "answer": answer,
        "sources": sources or [],
        "scope": f"{doc_category or ''}|{top_k}",
        "expire_at": time.time() + settings.semantic_cache_ttl,
    }

    client = get_redis()
    if client is not None:
        try:
            client.set(key, json.dumps(payload, ensure_ascii=False), ex=settings.semantic_cache_ttl)
            client.zadd(REDIS_INDEX_KEY, {key: time.time()})
            # 索引本身也设 TTL，防止无限增长
            client.expire(REDIS_INDEX_KEY, settings.semantic_cache_ttl * 2)
            return True
        except Exception as exc:
            logger.warning("写入 Redis 缓存失败，改用内存缓存: %s", exc)

    _MEMORY_CACHE.set(key, payload)
    return True


def clear_cache() -> int:
    """清空语义缓存（改完知识库或换 embedding 模型后必须清）。"""
    client = get_redis()
    removed = 0
    if client is not None:
        try:
            keys = client.zrange(REDIS_INDEX_KEY, 0, -1)
            if keys:
                removed += client.delete(*keys)
            client.delete(REDIS_INDEX_KEY)
        except Exception as exc:
            logger.warning("清空 Redis 缓存失败: %s", exc)
    removed += len(_MEMORY_CACHE)
    _MEMORY_CACHE.clear()
    logger.info("语义缓存已清空，删除 %s 项", removed)
    return removed


# TODO(第二批增强):
#   1. 用 RedisSearch/RediSearch 的向量索引做 ANN 检索，替代当前 O(N) 全量比对；
#   2. 缓存 key 加入 embedding 模型版本，换模型时自动失效；
#   3. 记录命中率指标（hits/misses）暴露到 /api/admin/stats。
