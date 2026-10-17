"""检索评估指标的单元测试：用构造好的检索结果验证 MRR / HitRate 计算正确。"""

from __future__ import annotations

from typing import Any

from app.core.evaluation.retriever_eval import evaluate_retrieval, load_golden_set


class _Node:
    def __init__(self, chunk_id: str, text: str = "内容"):
        self.metadata = {"chunk_id": chunk_id}
        self.node_id = chunk_id
        self._text = text

    def get_content(self) -> str:
        return self._text


class _Scored:
    def __init__(self, chunk_id: str, score: float = 1.0):
        self.node = _Node(chunk_id)
        self.score = score


class FakeRetriever:
    """按 question 返回预设顺序的结果。"""

    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping

    def retrieve(self, question: str) -> list[Any]:
        return [_Scored(cid) for cid in self.mapping.get(question, [])]


def test_mrr_and_hit_rate():
    retriever = FakeRetriever({"Q1": ["c9", "c1", "c2"], "Q2": ["c3", "c4"]})
    dataset = [
        {"question": "Q1", "ground_truth_ids": ["c1"]},   # 第 2 位命中 -> RR=0.5
        {"question": "Q2", "ground_truth_ids": ["c3"]},   # 第 1 位命中 -> RR=1.0
    ]
    result = evaluate_retrieval(retriever, dataset, top_k=5)
    assert result["metrics"]["mrr@5"] == 0.75
    assert result["metrics"]["hit_rate@5"] == 1.0
    assert result["dataset_size"] == 2


def test_miss_case():
    retriever = FakeRetriever({"Q1": ["x1", "x2"]})
    result = evaluate_retrieval(retriever, [{"question": "Q1", "ground_truth_ids": ["c1"]}], top_k=5)
    assert result["metrics"]["mrr@5"] == 0.0
    assert result["metrics"]["hit_rate@5"] == 0.0


def test_load_golden_set(work_tmp):
    p = work_tmp / "golden.jsonl"
    p.write_text(
        '{"question":"Q1","ground_truth_ids":["c1"]}\n\n{"question":"Q2","ground_truth_ids":["c2"]}\n',
        encoding="utf-8",
    )
    items = load_golden_set(p)
    assert len(items) == 2
