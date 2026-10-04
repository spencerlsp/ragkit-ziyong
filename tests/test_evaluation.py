"""评估层的测试。

这一层的测试有个特别之处：**指标是数学**，所以可以用手算出来的确切数值钉死，
不需要「大概对就行」。评估指标算错是最难发现也最致命的 bug ——
你会拿一个错误的数字去做技术决策。

运行器部分仍然全离线：用一个最小假客户端假装 Milvus。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ragkit.config import Settings
from ragkit.errors import EvaluationError
from ragkit.evaluation import EvalSample, dump_dataset, evaluate, load_dataset
from ragkit.evaluation.metrics import hit_rate, ndcg_at_k, recall_at_k, reciprocal_rank
from ragkit.indexing import MilvusIndexer
from ragkit.retrieval import Retriever

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "openai_embedding_dim": 4,
        "milvus_uri": "http://fake-milvus:19530",
        "milvus_collection": "ragkit_eval_test",
        "top_k": 3,
        "request_timeout": 5.0,
        "max_concurrency": 4,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_hit(chunk_id: str, score: float, *, doc_id: str = "d1") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "distance": score,
        "entity": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": f"{chunk_id} 的内容",
            "chunk_index": 0,
            "source": f"{doc_id}.txt",
        },
    }


class FakeMilvusClient:
    """评估用最小假客户端。

    ⚠️ 这是项目里第三个 FakeMilvusClient 了（前两个在 test_indexing /
    test_retrieval）。它们已经开始漂移 —— 下次抄漏一个方法就会像上次
    那样「测试莫名其妙报 AttributeError」。
    正经做法是抽到 tests/conftest.py 或 tests/fakes.py 里共用一个，
    但那需要同时改三个测试文件，而我这边跑不了 pytest 验证 ——
    所以先记着，等你能跑测试时我们一起做这个重构。
    """

    def __init__(self, hits: list[list[dict[str, Any]]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.collections: set[str] = set()
        self.hits = hits or []
        self.closed = False
        self.fail_on: str | None = None
        self.fail_first_search = False

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        self.calls.append((name, payload))

    async def has_collection(self, collection_name: str, **kwargs: Any) -> bool:
        self._record("has_collection", {"collection_name": collection_name})
        await asyncio.sleep(0)
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        self._record("create_collection", {"collection_name": collection_name, **kwargs})
        await asyncio.sleep(0)
        self.collections.add(collection_name)

    async def search(self, collection_name: str, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._record("search", {"collection_name": collection_name, **kwargs})
        await asyncio.sleep(0)
        if self.fail_first_search:
            self.fail_first_search = False
            raise RuntimeError("假装这条样本检索失败")
        if self.fail_on == "search":
            raise RuntimeError("假装所有检索都失败")
        return self.hits

    async def hybrid_search(
        self, collection_name: str, **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        self._record("hybrid_search", {"collection_name": collection_name, **kwargs})
        await asyncio.sleep(0)
        return self.hits

    async def close(self) -> None:
        self._record("close", {})
        self.closed = True


class StubEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.closed = False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.5] * self.dim for _ in texts]

    async def aclose(self) -> None:
        self.closed = True


def make_retriever(client: FakeMilvusClient, **overrides: object) -> Retriever:
    settings = make_settings(**overrides)
    return Retriever(
        embedder=StubEmbedder(dim=settings.openai_embedding_dim),
        indexer=MilvusIndexer(settings, client=client),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# 指标：全部用手算值钉死
# ---------------------------------------------------------------------------


def test_hit_rate_only_cares_about_top_k() -> None:
    """相关内容在第 3 名：看前 2 条没命中，看前 3 条命中。"""
    retrieved = ["a", "b", "c"]
    relevant = {"c"}
    assert hit_rate(retrieved, relevant, 2) == 0.0
    assert hit_rate(retrieved, relevant, 3) == 1.0


def test_hit_rate_miss_is_zero() -> None:
    """一条都没命中就是 0.0。"""
    assert hit_rate(["a", "b"], {"z"}, 2) == 0.0


def test_recall_counts_proportion_of_relevant_found() -> None:
    """retrieved 前 2 条命中 {b, d} 里的 1 个；前 4 条命中 2 个。"""
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 1.0


def test_recall_dedupes_retrieved_ids() -> None:
    """检索结果里有重复 ID 时不能重复计数，否则召回率会虚高。

    没有去重的话：前 3 条里有 c1 出现两次，会算成命中 2 个；
    去重后只算命中 1 个。
    """
    retrieved = ["c1", "c1", "c2"]
    relevant = {"c1", "c3"}
    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_recall_with_empty_relevant_is_zero() -> None:
    """没有标准答案时召回率无定义，约定返回 0.0（而不是除零崩溃）。"""
    assert recall_at_k(["a", "b"], set(), 2) == 0.0


def test_reciprocal_rank_first_position() -> None:
    """第一条就命中 → 1.0。"""
    assert reciprocal_rank(["a", "b"], {"a"}, 2) == 1.0


def test_reciprocal_rank_second_position() -> None:
    """第 2 名命中 → 1/2。"""
    assert reciprocal_rank(["a", "b", "c"], {"b"}, 3) == 0.5


def test_reciprocal_rank_miss() -> None:
    """前 k 条全没命中 → 0.0。"""
    assert reciprocal_rank(["a", "b"], {"z"}, 2) == 0.0


def test_reciprocal_rank_respects_k() -> None:
    """命中项在 k 之外就不算 —— 这是 k 的意义所在。"""
    assert reciprocal_rank(["a", "b", "c"], {"c"}, 2) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    """完美排序（相关内容全在最前面）→ 1.0。"""
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_penalizes_late_hits() -> None:
    """相关内容排到第 2 位时的折扣：1/log2(3)。

    手算：只有 a 相关且排在第 2 位。
      DCG  = 0/log2(2) + 1/log2(3) = 1/log2(3) ≈ 0.6309
      IDCG = 1/log2(2)             = 1.0        （理想情况它该排第 1）
      NDCG = 1/log2(3)
    """
    assert ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_with_empty_relevant_is_zero() -> None:
    """没有标准答案时 NDCG 无定义，约定返回 0.0。"""
    assert ndcg_at_k(["a", "b"], set(), 2) == 0.0


def test_ndcg_ideal_uses_k_as_upper_bound() -> None:
    """标准答案比 k 多时，IDCG 的上界是 k，不是 len(relevant)。

    这条防的是「分母虚大」：如果有 5 个正确答案、只看前 2 条，
    那么最好的情况也只能命中 2 个，IDCG 就该按 2 个算。
    用 5 去算 IDCG 的话，分数永远上不到 1.0。
    """
    # 前 2 条正好是 5 个正确答案里的 2 个 —— 这已经是 k=2 下的最好结果
    assert ndcg_at_k(["a", "b", "c", "d", "e"], {"a", "b", "c", "d", "e"}, 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------


def test_eval_sample_dedupes_ids_keeping_order() -> None:
    """relevant_ids 去重且保持原顺序（顺序稳定才能让 JSONL 文件 diff 稳定）。"""
    sample = EvalSample(question="q", relevant_ids=["b", "a", "b", "c", "a"])
    assert sample.relevant_ids == ["b", "a", "c"]


def test_eval_sample_rejects_blank_question() -> None:
    with pytest.raises(ValueError):
        EvalSample(question="   ", relevant_ids=["a"])


def test_dataset_roundtrip(tmp_path: Path) -> None:
    """写出去再读回来，内容必须一致。"""
    samples = [
        EvalSample(question="问题一", relevant_ids=["a", "b"]),
        EvalSample(question="问题二", relevant_ids=["c"], metadata={"level": "easy"}),
    ]
    target = tmp_path / "eval.jsonl"

    assert dump_dataset(samples, target) == 2

    loaded = load_dataset(target)
    assert [s.question for s in loaded] == ["问题一", "问题二"]
    assert loaded[0].relevant_ids == ["a", "b"]
    assert loaded[1].metadata == {"level": "easy"}


def test_load_dataset_reports_line_number(tmp_path: Path) -> None:
    """解析失败要带上行号 —— 否则 500 行的文件你只能一行行数。"""
    target = tmp_path / "bad.jsonl"
    target.write_text(
        '{"question": "好的一行", "relevant_ids": ["a"]}\n这不是 JSON\n',
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError) as exc_info:
        load_dataset(target)

    assert "2" in str(exc_info.value)


def test_load_dataset_reports_validation_line_number(tmp_path: Path) -> None:
    """JSON 合法但字段不合法时，也要带行号。"""
    target = tmp_path / "invalid.jsonl"
    target.write_text(
        '{"question": "ok", "relevant_ids": ["a"]}\n{"question": "  ", "relevant_ids": []}\n',
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError) as exc_info:
        load_dataset(target)

    assert "2" in str(exc_info.value)


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        load_dataset(tmp_path / "not-there.jsonl")


def test_load_dataset_skips_blank_lines(tmp_path: Path) -> None:
    """末尾的空行不该被当成坏数据。"""
    target = tmp_path / "trailing.jsonl"
    target.write_text(
        '{"question": "q1", "relevant_ids": ["a"]}\n\n\n',
        encoding="utf-8",
    )
    assert len(load_dataset(target)) == 1


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------


async def test_evaluate_aggregates_metrics() -> None:
    """三个样本、固定检索结果，手工核对每个聚合指标。

    检索结果恒为 [c1, c2, c3]，k=3：
      s1 relevant={c1} → hit 1, recall 1, rr 1,   ndcg 1.0
      s2 relevant={c3} → hit 1, recall 1, rr 1/3, ndcg 0.5
      s3 relevant={z}  → 全 0

    聚合：hit_rate=2/3, recall=2/3, mrr=(1+1/3)/3=4/9, ndcg=(1+0.5)/3=0.5
    """
    client = FakeMilvusClient([[make_hit("c1", 0.9), make_hit("c2", 0.8), make_hit("c3", 0.7)]])
    samples = [
        EvalSample(question="q1", relevant_ids=["c1"]),
        EvalSample(question="q2", relevant_ids=["c3"]),
        EvalSample(question="q3", relevant_ids=["zzz"]),
    ]

    report = await evaluate(samples, make_retriever(client), k=3)

    assert report.n_samples == 3
    assert report.n_failed == 0
    assert report.metrics["hit_rate"] == pytest.approx(2 / 3)
    assert report.metrics["recall"] == pytest.approx(2 / 3)
    assert report.metrics["mrr"] == pytest.approx(4 / 9)
    assert report.metrics["ndcg"] == pytest.approx(0.5)
    assert [r.question for r in report.results] == ["q1", "q2", "q3"]


async def test_evaluate_isolates_failures() -> None:
    """一个样本失败不该毁掉整轮评估 —— 这是 M8 的核心异步知识点。

    用 max_concurrency=1 让执行顺序确定：第一条失败，后两条正常。
    """
    client = FakeMilvusClient([[make_hit("c1", 0.9)]])
    client.fail_first_search = True
    samples = [
        EvalSample(question="q1", relevant_ids=["c1"]),
        EvalSample(question="q2", relevant_ids=["c1"]),
        EvalSample(question="q3", relevant_ids=["c1"]),
    ]

    report = await evaluate(samples, make_retriever(client), k=1, max_concurrency=1)

    assert report.n_samples == 3
    assert report.n_failed == 1
    assert len(report.results) == 2
    assert [f.question for f in report.failures] == ["q1"]
    assert "RuntimeError" in report.failures[0].error
    # 指标只对成功的两条求平均 —— 它们是满分
    assert report.metrics["hit_rate"] == pytest.approx(1.0)


async def test_evaluate_all_failed_returns_zero_metrics() -> None:
    """全部失败时不抛异常，而是返回一份「指标 0、失败 3」的报告。

    看指标之前先看 n_failed —— 这是报告里为什么要有这个字段。
    """
    client = FakeMilvusClient([[make_hit("c1", 0.9)]])
    client.fail_on = "search"
    samples = [EvalSample(question=f"q{i}", relevant_ids=["c1"]) for i in range(3)]

    report = await evaluate(samples, make_retriever(client), k=1)

    assert report.n_failed == 3
    assert report.results == []
    assert report.metrics["hit_rate"] == 0.0


async def test_evaluate_rejects_empty_dataset() -> None:
    """空数据集一定是配置错了，要报错而不是返回一份全是 0 的报告。"""
    client = FakeMilvusClient()
    with pytest.raises(EvaluationError):
        await evaluate([], make_retriever(client))


async def test_evaluate_passes_mode_to_retriever() -> None:
    """mode="hybrid" 要一路传到 indexer —— 评估器不能把它吞掉。"""
    client = FakeMilvusClient([[make_hit("c1", 0.9)]])
    samples = [EvalSample(question="q", relevant_ids=["c1"])]

    report = await evaluate(samples, make_retriever(client), k=1, mode="hybrid")

    assert report.n_failed == 0
    assert "hybrid_search" in client.names()


async def test_evaluate_matches_on_doc_id() -> None:
    """标注只有文档级时，可以按 doc_id 比对。"""
    client = FakeMilvusClient([[make_hit("c1", 0.9, doc_id="doc-x")]])
    samples = [EvalSample(question="q", relevant_ids=["doc-x"])]

    by_chunk = await evaluate(samples, make_retriever(client), k=1, match_on="chunk_id")
    by_doc = await evaluate(samples, make_retriever(client), k=1, match_on="doc_id")

    assert by_chunk.metrics["hit_rate"] == 0.0  # chunk_id 是 c1，对不上 doc-x
    assert by_doc.metrics["hit_rate"] == 1.0


async def test_evaluate_report_summary_is_readable() -> None:
    """summary() 要能一眼看出效果和失败情况。"""
    client = FakeMilvusClient([[make_hit("c1", 0.9)]])
    samples = [EvalSample(question="q", relevant_ids=["c1"])]

    report = await evaluate(samples, make_retriever(client), k=1)
    text = report.summary()

    assert "dense" in text
    assert "k=1" in text
    assert "ndcg" in text
