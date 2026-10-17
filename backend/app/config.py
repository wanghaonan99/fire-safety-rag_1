"""全局配置：用 pydantic-settings 从环境变量 / .env 读取，集中管理、禁止散落硬编码。

设计要点：
1. 所有可调参数都在这里，替换 embedding 模型或 LLM 只改环境变量，不动业务代码；
2. 提供若干 `@property` 做格式派生（如把 DATABASE_URL 转成 SQLAlchemy 方言串）；
3. `get_settings()` 带 lru_cache，全进程只解析一次。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目全部配置项。字段名大小写不敏感，对应 .env 中的大写变量。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # 忽略 .env 里多余变量，避免部署环境报错
        case_sensitive=False,
    )

    # ---------- 运行环境 ----------
    app_name: str = "fire-safety-rag"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = True
    api_prefix: str = "/api"
    log_level: str = "INFO"

    # ---------- LLM：DeepSeek ----------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = 0.1
    deepseek_max_tokens: int = 1024

    # ---------- Embedding ----------
    embedding_provider: Literal["local", "siliconflow"] = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024          # bge-m3 稠密向量维度，换模型必须同步改
    # auto = 有可用 CUDA 就用 GPU，否则 CPU；也可显式写 cpu / cuda / cuda:0
    embedding_device: str = "auto"
    # 批大小：CPU 建议 16，GPU 可以调到 64~128（显存够的话吞吐高很多）
    embedding_batch_size: int = 16
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    # ---------- Rerank ----------
    rerank_provider: Literal["local", "none"] = "local"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 5
    # fp16 只在 GPU 上有意义（CPU 上反而更慢）；最终取「本值 or 是否检测到 CUDA」
    rerank_use_fp16: bool = False

    # ---------- 数据库 ----------
    database_url: str = ""             # Neon 的 -pooler 连接串
    db_echo: bool = False
    db_pool_size: int = 5
    # 语句超时（毫秒）。⚠️ 用 pooler 连接串时该值不会通过启动参数下发
    # （PgBouncer 不允许），需要时在事务内 SET LOCAL 生效
    db_statement_timeout_ms: int = 30000

    # ---------- 缓存 ----------
    redis_url: str = ""
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    semantic_cache_ttl: int = 86400

    # ---------- 检索 ----------
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieve_top_k: int = 10
    rerank_top_k: int = 5
    similarity_top_k: int = 10
    bm25_top_k: int = 10
    # ⚠️ Step-back 查询改写默认关闭：它是"再问一次 DeepSeek 生成上位问题"，
    #    然后**再完整检索一遍**。实测代价 = 一次 LLM(2.5s) + 一次检索(3.4~11s)，
    #    而跨区域数据库（Neon 新加坡）下单次检索就要好几秒，性价比很低。
    #    等数据库本地化后再打开收益更明显。
    enable_query_transform: bool = False
    enable_hyde: bool = False
    enable_hype: bool = True
    # 融合检索是否走异步并发。
    # ⚠️ 默认 False：本项目整条链路是同步的（FastAPI 同步处理器 + psycopg2 + 同步 httpx），
    #    打开后 QueryFusionRetriever 会用 asyncio.run 起临时事件循环、并让 PGVectorStore
    #    走 asyncpg，一旦有第二次检索就会遇到
    #    "Event loop is closed" 继而 "Detected nested async"。
    #    将来把路由与检索整体改成 async（aquery/aretrieve/achat）再打开它换并发。
    retrieval_use_async: bool = False

    # ---------- 会话 ----------
    session_backend: Literal["postgres", "redis"] = "postgres"
    session_max_turns: int = 6

    # ---------- 上传 ----------
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 50

    # ---------- 解析质量 ----------
    # PDF 文字层质量分低于该值即判定为"乱码/扫描件"，改用 PyMuPDF 或降级 OCR
    # 正常中文文档约 0.35~0.60；方正书版乱码 PDF < 0.05
    min_text_quality: float = 0.15
    # 入库前丢弃低信息密度段落（封面/目次/前言/起草人/参考文献）。
    # 它们会被检索召回成噪声（实测 top-5 里出现过封面块与参考文献块），
    # 也是"从模板句出废题"的根源。消融实验可关掉它，用「噪声率」指标量化收益。
    enable_chunk_filter: bool = True
    # 拼进提示词的单个切块最大字符数。
    # 原来是 1200，实测会把 GB 51283 表4.1.5 的防火间距表数值切掉（表格从第 934 字符开始），
    # 导致明明库里有数值、模型却回答"需查表"。含表格的切块另有更宽松的上限（见 prompt.py）。
    prompt_node_max_chars: int = 1800

    # ---------- 模型缓存目录 ----------
    # ⚠️ 故意**不叫** HF_HOME：本字段是 pydantic 字段，环境变量优先级高于 .env，
    #    如果叫 hf_home，机器上那个把 HF_HOME 设成不可写路径（如盘符根目录下）的
    #    环境变量就会悄悄覆盖这里的配置。用独立名字，再由 apply_hf_cache_env()
    #    统一、强制地把 HF_HOME/HF_HUB_CACHE/LLAMA_INDEX_CACHE_DIR 指向可写目录。
    # 设成空字符串表示跳过该项（仍会尝试下一个候选目录）。
    model_cache_dir: str = "./data/indexes/hf_cache"

    # ---------- CORS ----------
    cors_origins: str = "http://localhost:3000"

    # ---------- 管理鉴权 ----------
    # ⚠️ 上生产前必须改成强随机串（默认值只用于本地开发）
    admin_token: str = "change-me"

    # ---------- LangFuse ----------
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---------- LangFuse 兼容别名（部分 SDK 读取大写键） ----------
    @field_validator("deepseek_base_url", "siliconflow_base_url", "langfuse_host", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/") if isinstance(v, str) else v

    # ---------- 派生属性 ----------
    @property
    def cors_origin_list(self) -> list[str]:
        """把逗号分隔的 CORS_ORIGINS 解析成列表。"""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy 2.x 用的驱动串；Neon 给的 postgresql:// 显式指定 psycopg2。"""
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例配置。测试里可用 get_settings.cache_clear() 重置。"""
    return Settings()


def apply_hf_cache_env() -> str:
    """把模型缓存相关环境变量强制指向**确认可写**的目录，返回最终路径。

    为什么必须"强制 + 永不失败"：
    1. huggingface_hub / llama-index 读的是进程环境变量，pydantic 配置管不到；
       机器上若把 HF_HOME 设成普通用户没有写权限的路径（常见于设在盘符根目录下），
       所有模型下载都会以 `PermissionError ... Check cache directory permissions` 失败。
    2. llama_index 的 `get_cache_dir()` 在拿不到 cache_folder 时会去创建
       `%LOCALAPPDATA%\\llama_index`；该目录若被权限问题污染，会直接
       `PermissionError: [WinError 5]`，把 embedding 加载整个带崩。
       所以我们**总是**显式给出 cache_folder（见 embedder.py），并在这里也设好
       LLAMA_INDEX_CACHE_DIR 兜住其它内部调用。
    """
    global _HF_CACHE_PATH

    import os
    import tempfile
    from pathlib import Path

    if _HF_CACHE_PATH:
        return _HF_CACHE_PATH

    # 依次尝试：显式配置 -> 项目内默认 -> 系统临时目录（最后一个几乎必然可写）
    candidates = [settings.model_cache_dir, "./data/indexes/hf_cache", tempfile.gettempdir()]
    chosen: Path | None = None
    failures: list[str] = []

    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".dsh_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{raw}（{type(exc).__name__}）")
            continue
        chosen = candidate
        break

    if chosen is None:
        print(f"[config] 警告：找不到可写的模型缓存目录：{'; '.join(failures)}")
        return ""

    if failures:
        print(f"[config] 提示：已跳过不可写的缓存目录 {', '.join(failures)}，改用 {chosen}")

    os.environ["HF_HOME"] = str(chosen)
    os.environ["HF_HUB_CACHE"] = str(chosen / "hub")
    # 防止 llama_index 内部再去创建 %LOCALAPPDATA%\llama_index 而踩到权限问题
    llama_index_dir = chosen / "llama_index_base"
    try:
        llama_index_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001  建不出来也不影响：embedder 会显式传 cache_folder
        pass
    os.environ["LLAMA_INDEX_CACHE_DIR"] = str(llama_index_dir)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(chosen))
    _HF_CACHE_PATH = str(chosen)
    return _HF_CACHE_PATH


_HF_CACHE_PATH: str | None = None

settings = get_settings()
# 导入即纠正缓存位置（幂等、失败只告警不抛错）
apply_hf_cache_env()
