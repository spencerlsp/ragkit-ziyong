"""评估跑批：把评估集跑一遍，算指标，出报告。

★ M8 的新异步知识点：``asyncio.gather(..., return_exceptions=True)``

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
        """一行摘要，方便打日志和横向对比。已经写好了。"""
        head = f"[{self.mode}] k={self.k} n={self.n_samples}"
        if self.n_failed:
            head += f" 失败={self.n_failed}"
        body = "  ".join(f"{name}={value:.3f}" for name, value in self.metrics.items())
        return f"{head}  {body}"


def _extract_id(hit: ScoredChunk, match_on: str) -> str:
    """按 match_on 决定用哪个字段和标准答案比对。已经写好了。"""
    return hit.chunk.doc_id if match_on == "doc_id" else hit.chunk.chunk_id


async def evaluate(
    samples: Sequence[EvalSample],
    retriever: Retriever,
    *,
    k: int = 5,
    mode: Literal["dense", "hybrid"] = "dense",
    match_on: Literal["chunk_id", "doc_id"] = "chunk_id",
    max_concurrency: int = 8,
) -> EvalReport:
    """跑一遍评估集，返回完整报告。

    TODO(你)：七步。

        1) 拦住空数据集：

               if not samples:
                   raise EvaluationError("评估集是空的，没有可评估的样本")

           为什么报错而不是返回空报告：空评估集一定是配置错了
           （文件路径写错、加载漏了）。返回一份「所有指标都是 0」的报告，
           你会以为「效果很差」，而不是「我根本没跑起来」。

        2) 建闸门：``sem = asyncio.Semaphore(max_concurrency)``
           —— 和 M4/M5 一样。评估会打真数据库，并发要保守。

        3) 定义单样本的协程：

               async def one(sample: EvalSample) -> SampleResult:
                   async with sem:
                       hits = await retriever.retrieve(
                           sample.question, top_k=k, mode=mode
                       )
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

           注意闸门只包住 retrieve（真正打网络/数据库的那段），
           打分是纯计算，不进闸门。

        4) **并行跑，且容错**：

               outcomes = await asyncio.gather(
                   *(one(sample) for sample in samples), return_exceptions=True
               )

           ``return_exceptions=True`` 是关键：不加它，第一个异常就会让
           gather 把异常抛出来，剩下所有样本白跑了。
           加了它，失败样本的**异常对象本身**会被放进返回值列表里。

        5) 分流。这一步最容易漏 —— 返回列表里混着两种东西：

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

           * ``zip(..., strict=True)`` 保证结果和样本一一对应
             （gather **保证返回顺序和传入顺序一致**，这点和 TaskGroup 不同）。
           * 判断 ``isinstance(outcome, BaseException)`` 而不是 ``Exception``：
             ``CancelledError`` 继承自 BaseException 而不是 Exception，
             用 Exception 会漏掉被取消的样本，然后它在 else 分支里
             被当成正常结果塞进 results —— 后果是后面取属性时莫名其妙报错。

        6) 算平均。写个小工具函数：

               def mean(values: list[float]) -> float:
                   return sum(values) / len(values) if values else 0.0

               metrics = {
                   "hit_rate": mean([r.hit_rate for r in results]),
                   "recall": mean([r.recall for r in results]),
                   "mrr": mean([r.rr for r in results]),        # RR 的平均就是 MRR
                   "ndcg": mean([r.ndcg for r in results]),
               }

           **指标只对成功的样本求平均。** 把失败的样本按 0 分计入，
           会把「接口偶尔超时」和「检索质量差」混为一谈。
           失败率应该单独看 —— 所以报告里有 ``n_failed``。

           全部失败时 metrics 全是 0.0。这是个有歧义的结果，
           所以**看指标之前先看 n_failed**。

        7) 组装并返回 EvalReport（把 k / mode / match_on /
           n_samples / n_failed / metrics / results / failures 都填上）。
    """
    # 1) 拦住空数据集
    if not samples:
        raise EvaluationError("评估集是空的，没有可评估的样本")

    # 2) 并发闸门，控制最大并发数
    sem = asyncio.Semaphore(max_concurrency)

    # 3) 单样本协程
    async def one(sample: EvalSample) -> SampleResult:
        async with sem:
            # 真正打数据库/检索，放在闸门内
            hits = await retriever.retrieve(sample.question, top_k=k, mode=mode)
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
