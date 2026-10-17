"""Embedding 模型构建（本地 HuggingFace 或 SiliconFlow API 二选一）。

换模型的唯一改动点：
1. 改 .env 里的 EMBEDDING_MODEL / EMBEDDING_DIM / EMBEDDING_PROVIDER；
2. ⚠️ 维度变化必须**重建向量表**（pgvector 列维度固定），见 scripts/init_db.sql 注释。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from app.config import apply_hf_cache_env, settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device() -> str:
    """决定 embedding 跑在哪：auto -> 有 CUDA 用 cuda，否则 cpu。

    常见"明明有显卡却还在用 CPU"的原因：装的是 **CPU 版 torch**
    （版本号形如 `2.x.x+cpu`，`torch.version.cuda is None`）。
    需要换成 CUDA 版：pip install torch --index-url https://download.pytorch.org/whl/cu128
    """
    want = (settings.embedding_device or "auto").lower()
    try:
        import torch
    except ImportError:
        return "cpu"

    if want != "auto":
        if want.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("配置要求 %s 但当前 torch 不支持 CUDA（是否为 +cpu 版本？），回退 CPU", want)
            return "cpu"
        return want

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("检测到 CUDA 设备：%s（%.1fGB），embedding 将跑在 GPU 上", name, total_gb)
        return "cuda"
    if torch.version.cuda is None:
        logger.warning("torch 是 CPU 版（torch.version.cuda is None），embedding 只能用 CPU；"
                       "要启用显卡请改装 CUDA 版 torch，见 README「用上显卡」")
    return "cpu"


def build_embed_model() -> Any:
    """按 EMBEDDING_PROVIDER 返回 LlamaIndex 的 Embedding 对象。"""
    if settings.embedding_provider == "siliconflow":
        return _build_siliconflow_embedding()
    return _build_local_embedding()


def find_cached_model_path(model_name: str, extra_roots: list[str] | None = None) -> str | None:
    """在 HF 缓存里找一个**文件齐全**的 snapshot 目录，找不到返回 None。

    为什么要这个：直接把仓库名（如 BAAI/bge-m3）交给 sentence-transformers / transformers 时，
    它们会先向 Hub 发 HEAD 请求确认版本。**断网时这条路径会走进异常分支**，
    报出极具误导性的
    `Unrecognized processing class in BAAI/bge-m3. Can't instantiate a processor, a tokenizer...`
    ——明明本地缓存是完整的。
    另外，缓存里可能残留"下载中断"的残缺快照（只有 model.safetensors、没有 tokenizer），
    也会触发同样的报错。这里只认"配置 + 权重 + 分词器"三件套都在的快照。

    附带好处：直接给本地路径后不再有 Hub 往返，模型加载更快也更确定。
    """
    from pathlib import Path

    roots: list[str] = []
    if extra_roots:
        roots.extend(extra_roots)

    # 项目自己的缓存根：embedding 走 llama-index 的子目录，reranker/FlagEmbedding 走 hub
    cache_root = apply_hf_cache_env()
    if cache_root:
        roots.append(str(Path(cache_root) / "llama_index"))
        roots.append(str(Path(cache_root) / "hub"))

    for env_key in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_key)
        if value:
            roots.append(value if env_key == "HF_HUB_CACHE" else str(Path(value) / "hub"))
    roots.append(str(Path.home() / ".cache" / "huggingface" / "hub"))

    repo_dir = "models--" + model_name.replace("/", "--")
    weights = ("model.safetensors", "pytorch_model.bin")
    tokenizers = ("tokenizer.json", "sentencepiece.bpe.model", "spm.model", "vocab.txt")

    for root in roots:
        snapshots = Path(root) / repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = [d for d in snapshots.iterdir() if d.is_dir()]
        # 优先最近修改的（通常是最近一次成功加载过的）
        for snap in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            has_config = (snap / "config.json").is_file()
            has_weights = any((snap / w).is_file() for w in weights)
            has_tokenizer = any((snap / t).is_file() for t in tokenizers)
            if has_config and has_weights and has_tokenizer:
                return str(snap)
    return None


def _build_local_embedding() -> Any:
    """本地 HuggingFace 加载 BAAI/bge-m3（首次会下载约 2.2GB）。"""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    device = resolve_device()
    # GPU 上可以用大 batch 提升吞吐；CPU 上大 batch 收益很小反而更吃内存
    batch_size = settings.embedding_batch_size
    if device == "cuda" and batch_size < 32:
        batch_size = 64
        logger.info("GPU 模式，embedding 批大小自动提到 %s", batch_size)

    cache_root = apply_hf_cache_env()
    cache_folder = None
    if cache_root:
        from pathlib import Path

        cache_folder = str(Path(cache_root) / "llama_index")

    # 优先用本地完整快照，避免断网时的 Hub 查询踩坑（详见 find_cached_model_path 注释）
    local_path = find_cached_model_path(
        settings.embedding_model,
        extra_roots=[cache_folder] if cache_folder else None,
    )

    kwargs: dict[str, Any] = {
        "model_name": local_path or settings.embedding_model,
        # bge 系列推荐把 query 侧指令加上，提升短问句对长文档的召回
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
        "embed_batch_size": batch_size,
        "max_length": 512,
        "device": device,
        # ⚠️ 不要设 trust_remote_code=True：bge-m3 是标准 XLM-R 结构，不需要它，
        #    而开启后 sentence-transformers 可能弹出"是否执行远程代码"的交互确认，
        #    在非交互环境（后台任务/容器）里会**永久挂起**且不报错。
        "trust_remote_code": False,
    }

    # 固定模型缓存位置：HuggingFaceEmbedding 默认用 llama-index 自己的缓存目录，
    # 不受 HF_HOME 影响，换个启动方式（终端 vs 后台任务）就会各下一份 4GB+ 的模型。
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
        logger.info("Embedding 缓存目录：%s", cache_folder)

    logger.info(
        "加载本地 Embedding 模型 %s（device=%s, batch=%s）",
        "本地快照 " + local_path if local_path else settings.embedding_model + "（本地无缓存，将联网下载）",
        device,
        batch_size,
    )
    try:
        return HuggingFaceEmbedding(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if not local_path:
            raise
        # 本地快照加载失败（例如版本不兼容）时，退回按仓库名加载
        logger.warning("本地快照加载失败（%s），改为按仓库名加载 %s", exc, settings.embedding_model)
        kwargs["model_name"] = settings.embedding_model
        return HuggingFaceEmbedding(**kwargs)


def _build_siliconflow_embedding() -> Any:
    """走 SiliconFlow 的 OpenAI 兼容接口（Render 免费套餐推荐）。"""
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding

    if not settings.siliconflow_api_key:
        raise ValueError("EMBEDDING_PROVIDER=siliconflow 时必须配置 SILICONFLOW_API_KEY")

    logger.info("使用 SiliconFlow Embedding API: %s", settings.embedding_model)
    return OpenAILikeEmbedding(
        model_name=settings.embedding_model,
        api_key=settings.siliconflow_api_key,
        api_base=settings.siliconflow_base_url,
        embed_batch_size=16,
        # 部分兼容实现不支持 dimensions 参数，需要时再打开
        # additional_kwargs={"encoding_format": "float"},
    )


@lru_cache(maxsize=1)
def embed_texts(texts: tuple[str, ...]) -> list[list[float]]:
    """给非 LlamaIndex 代码用的批量向量化便捷函数（走单例模型）。"""
    from app.dependencies import embed_model_dep

    model = embed_model_dep()
    return [model.get_text_embedding(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度，供语义缓存比对用（避免多引一个依赖）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
