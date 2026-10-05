"""评估跑批：把评估集跑一遍，算指标，出报告。

M8 的新异步知识点：``asyncio.gather(..., return_exceptions=True)``

    注意它和 M2 的 TaskGroup 是**相反**的取舍：

    * 批量导入文件（M2）：一个文件坏了，整批没意义 → 快速失败，全取消。
    * 批量评估（这里）：一个样本失败只说明这一条有问题，
      剩下 99 条的结果依然有价值 → **容错继续**，最后把失败列出来。

    用 gather + return_exceptions=True，失败的协程不会中断别人，
    而是把**异常对象本身**放进结果列表里返回。
    所以拿到结果后必须逐个判断类型 —— 这是个容易忘的步骤。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import EvaluationError
from ..retrieval import Retriever
from ..schemas import ScoredChunk
from .dataset import EvalSample
from .metrics import hit_rate, ndcg_at_k, recall_at_k, reciprocal_rank

__all__ = ["EvalReport", "SampleResult", "FailureRecord", "evaluate"]


class SampleResult(BaseModel):
    """单个样本的检索结果和它的各项得分。

    注意 ScoredChunk 是 frozen 的，但**这个模型不是** ——
    合适就好，不必处处冻结。冻结的价值在于「传出去以后不该被改」，
    而这里是一次性算完就返回的报表数据。
    """

    question: str
    retrieved_ids: list[str]
    scores: list[float]
    hit_rate: float
    recall: float
    rr: float
    ndcg: float


class FailureRecord(BaseModel):
    """一条失败的样本。保留它比丢掉它有用得多。"""

    question: str
    error: str


class EvalReport(BaseModel):
    """一次评估的完整结果。"""

    model_config = ConfigDict(frozen=True)

    k: int
    mode: str
    match_on: str
    n_samples: int
    n_failed: int
    metrics: dict[str, float] = Field(default_factory=dict)
    results: list[SampleResult] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)

    def summary(self) -> str:
        """一行摘要，方便打日志和横向对比。"""
        head = f"[{self.mode}] k={self.k} n={self.n_samples}"
        if self.n_failed:
            head += f" 失败={self.n_failed}"
        body = "  ".join(f"{name}={value:.3f}" for name, value in self.metrics.items())
        return f"{head}  {body}"


def _extract_id(hit: ScoredChunk, match_on: str) -> str:
    """按 match_on 决定用哪个字段和标准答案比对。"""
    return hit.chunk.doc_id if match_on == "doc_id" else hit.chunk.chunk_id


async def evaluate(
    samples: Sequence[EvalSample],
    retriever: Retriever,
    *,
    k: int = 5,
    mode: Literal["dense", "hybrid"] = "dense",
    rerank: bool = False,
    match_on: Literal["chunk_id", "doc_id"] = "chunk_id",
    max_concurrency: int = 8,
) -> EvalReport:
    """跑一遍评估集，返回完整报告。"""
    # 1) 拦住空数据集
    if not samples:
        raise EvaluationError("评估集是空的，没有可评估的样本")

    # 2) 并发闸门，控制最大并发数
    sem = asyncio.Semaphore(max_concurrency)

    # 3) 单样本协程
    async def one(sample: EvalSample) -> SampleResult:
        async with sem:
            # 真正打数据库/检索，放在闸门内
            hits = await retriever.retrieve(sample.question, top_k=k, mode=mode, rerank=rerank)
        # 指标计算：纯CPU计算，不在sem内
        ids = [_extract_id(hit, match_on) for hit in hits]
        return SampleResult(
            question=sample.question,
            retrieved_ids=ids,
            scores=[hit.score for hit in hits],
            hit_rate=hit_rate(ids, sample.relevant_ids, k),
            recall=recall_at_k(ids, sample.relevant_ids, k),
            rr=reciprocal_rank(ids, sample.relevant_ids, k),
            ndcg=ndcg_at_k(ids, sample.relevant_ids, k),
        )

    # 4) 并发执行，return_exceptions=True 容错
    tasks = (one(sample) for sample in samples)
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    # 5) 分流：成功结果 / 失败记录
    results: list[SampleResult] = []
    failures: list[FailureRecord] = []
    for sample, outcome in zip(samples, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            failures.append(
                FailureRecord(
                    question=sample.question,
                    error=f"{type(outcome).__name__}: {outcome}",
                )
            )
        else:
            results.append(outcome)

    # 6) 求均值工具函数，只对成功样本计算指标
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "hit_rate": mean([r.hit_rate for r in results]),
        "recall": mean([r.recall for r in results]),
        "mrr": mean([r.rr for r in results]),
        "ndcg": mean([r.ndcg for r in results]),
    }

    # 7) 组装EvalReport返回
    report = EvalReport(
        k=k,
        mode=mode,
        match_on=match_on,
        n_samples=len(samples),
        n_failed=len(failures),
        metrics=metrics,
        results=results,
        failures=failures,
    )
    return report
