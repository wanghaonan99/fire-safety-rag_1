"""RAGAS 评估（第二批）：生成侧质量。

三个核心指标：
- **context_precision**：检索到的上下文里，有多少是真正相关的（信噪比）；
- **context_recall**：标准答案所需信息，检索上下文覆盖了多少（漏没漏）；
- **faithfulness**：回答是否**完全由上下文支撑**（消防场景最关键，防幻觉）。

RAGAS 用 LLM 当裁判，会消耗大量 DeepSeek token：20 条样本 × 3 指标约 60 次调用。
建议先用小样本（10-20 条）建立基线，调优时只跑变化的部分。

⚠️ RAGAS 依赖 langchain 的 LLM/Embedding 适配器，需要额外安装：
    pip install langchain-openai langchain-community
"""

from __future__ import annotations

import time
from typing import Any

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = ["context_precision", "context_recall", "faithfulness"]


def _patch_ragas_vertexai_import() -> None:
    """绕过 ragas 与 langchain-community 的版本断裂（否则 `import ragas` 直接失败）。

    ragas 0.4.x 的 `ragas/llms/base.py` 仍然写着
        from langchain_community.chat_models.vertexai import ChatVertexAI
    而该模块在 langchain-community 0.4 里已被移除，于是整个包无法导入。

    本项目全程用 DeepSeek 的 ChatOpenAI 当裁判，**永远用不到 VertexAI**，
    所以注册一个占位模块让 import 通过即可（不实现任何逻辑）。
    """
    import sys
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: D401 - 占位符，只用于满足导入
        """VertexAI 占位类：本项目不使用，实例化即报错以便及早暴露误用。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("VertexAI 未启用（本项目裁判模型为 DeepSeek）")

    shim.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[module_name] = shim
    logger.info("已为 ragas 注入 langchain_community.chat_models.vertexai 占位模块（版本兼容）")


def _build_ragas_judge():
    """RAGAS 的裁判 LLM / Embedding（用 DeepSeek + bge-m3，与线上一致才可比）。"""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    base = settings.deepseek_base_url.rstrip("/")
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=base if base.endswith("/v1") else f"{base}/v1",
        temperature=0.0,
    )

    if settings.embedding_provider == "siliconflow":
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
        )
    else:
        # 本地 embedding 通过 langchain-huggingface 适配。
        # ⚠️ 必须指向**本地快照目录**：直接给 "BAAI/bge-m3" 会让它去 huggingface.co
        #    拉取元数据，国内网络会长时间卡住（这正是主链路 find_cached_model_path 存在的原因）。
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            from app.core.ingestion.embedder import find_cached_model_path

            local = find_cached_model_path(settings.embedding_model)
            embeddings = HuggingFaceEmbeddings(
                model_name=str(local) if local else settings.embedding_model,
                cache_folder=settings.model_cache_dir,
            )
        except ImportError:
            logger.warning("未安装 langchain-huggingface，退回 OpenAI 兼容 embedding")
            embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.siliconflow_api_key,
                base_url=settings.siliconflow_base_url,
            )
    return llm, embeddings


def _build_metrics(names: list[str], llm: Any, embeddings: Any) -> list[Any]:
    """按名字构造 RAGAS 指标对象（不同版本导入路径有差异，这里做了兼容）。"""
    from ragas import metrics as ragas_metrics

    registry = {
        "context_precision": getattr(ragas_metrics, "context_precision", None),
        "context_recall": getattr(ragas_metrics, "context_recall", None),
        "faithfulness": getattr(ragas_metrics, "faithfulness", None),
        "answer_relevancy": getattr(ragas_metrics, "answer_relevancy", None),
    }
    built: list[Any] = []
    for name in names:
        metric = registry.get(name)
        if metric is None:
            logger.warning("RAGAS 指标不存在（版本差异），跳过：%s", name)
            continue
        # ragas>=0.2 的指标是带 llm/embeddings 的单例，直接可用；旧版需要实例化类
        try:
            built.append(metric)          # 新版：已初始化的单例
        except Exception:                 # noqa: BLE001
            built.append(metric(llm=llm, embeddings=embeddings))
    return built


def evaluate_generation(
    dataset: list[dict[str, Any]],
    metrics: list[str] | None = None,
    run_name: str = "ragas-baseline",
) -> dict[str, Any]:
    """跑 RAGAS 评估。

    dataset 每条须含：
      question / answer / contexts(list[str]) / ground_truth
    其中 contexts 由本项目的检索链路产出（见 app/api/routes_eval.py 的组装逻辑）。
    """
    _patch_ragas_vertexai_import()

    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate

    if not dataset:
        raise ValueError("评估集为空")

    metric_names = metrics or DEFAULT_METRICS
    llm, embeddings = _build_ragas_judge()
    built_metrics = _build_metrics(metric_names, llm, embeddings)

    hf_dataset = Dataset.from_list(dataset)
    start = time.perf_counter()
    logger.info("开始 RAGAS 评估：%s 条 × %s 个指标", len(dataset), len(built_metrics))

    result = ragas_evaluate(
        dataset=hf_dataset,
        metrics=built_metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,     # 单条失败不打断整体
        show_progress=True,
    )
    duration = time.perf_counter() - start

    # ragas 0.2 用 to_pandas()，旧版直接 dict()
    try:
        scores: dict[str, float] = {k: round(float(v), 4) for k, v in result.items()}
    except Exception:  # noqa: BLE001
        df = result.to_pandas()
        scores = {c: round(float(df[c].mean()), 4) for c in df.columns if df[c].dtype.kind == "f"}

    logger.info("RAGAS 完成：%s（%.1fs）", scores, duration)

    # 逐条得分：只给均值无法定位"是哪些题、哪类题拉低了指标"，
    # 而对 faithfulness 这种红线指标，定位问题样本比看均值重要得多。
    detail = _extract_item_scores(result, metric_names)
    if detail:
        bad = [d for d in detail if d["scores"].get("faithfulness") is not None
               and d["scores"]["faithfulness"] < 0.5]
        logger.info("逐条得分导出 %s 条；其中 faithfulness<0.5 的有 %s 条", len(detail), len(bad))

    return {
        "run_name": run_name,
        "eval_type": "generation",
        "metrics": scores,
        "dataset_size": len(dataset),
        "duration_sec": round(duration, 2),
        "detail": detail,
    }


def _extract_item_scores(result: Any, metric_names: list[str]) -> list[dict[str, Any]]:
    """从 RAGAS 结果里导出逐条得分（兼容 0.2 的旧列名与 0.4 的新列名）。"""
    try:
        df = result.to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法导出逐条得分：%s", exc)
        return []

    def pick(row: Any, *names: str) -> Any:
        for n in names:
            if n in row and row[n] is not None:
                return row[n]
        return None

    items: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        scores: dict[str, float | None] = {}
        for m in metric_names:
            value = pick(row, m)
            try:
                scores[m] = None if value is None or value != value else round(float(value), 4)
            except (TypeError, ValueError):
                scores[m] = None
        contexts = pick(row, "retrieved_contexts", "contexts") or []
        items.append(
            {
                "question": str(pick(row, "user_input", "question") or "")[:120],
                "answer": str(pick(row, "response", "answer") or "")[:500],
                "n_contexts": len(contexts) if hasattr(contexts, "__len__") else 0,
                "scores": scores,
            }
        )
    return items


def build_ragas_dataset_from_pipeline(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把"问题 + 检索上下文 + 生成答案"整理成 RAGAS 需要的扁平结构。

    items: [{question, answer, contexts: list[str], ground_truth}]
    TODO(第二批): 加 answer_relevancy 与 noise_sensitivity 两个指标，
                  并把 LangFuse trace_id 写进 eval_records.detail 便于回看。
    """
    rows: list[dict[str, Any]] = []
    for it in items:
        rows.append(
            {
                "question": it["question"],
                "answer": it.get("answer", ""),
                "contexts": it.get("contexts", []),
                "ground_truth": it.get("ground_truth", ""),
            }
        )
    return rows
