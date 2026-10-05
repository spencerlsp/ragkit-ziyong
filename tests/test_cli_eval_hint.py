"""``eval`` 那条诊断提示的测试。

单独一个文件是因为它只测一个纯函数（``_eval_mismatch_hint``），
不用起 CLI、不用连数据库 —— 构造几个 ``EvalReport`` 就够了。
"""

from __future__ import annotations

from ragkit.cli import _eval_mismatch_hint
from ragkit.evaluation import EvalReport, SampleResult


def sample(rr: float) -> SampleResult:
    hit = 1.0 if rr > 0 else 0.0
    return SampleResult(
        question="q",
        retrieved_ids=["c1"],
        scores=[0.9],
        hit_rate=hit,
        recall=hit,
        rr=rr,
        ndcg=hit,
    )


def report(*, rr: list[float], failed: int = 0) -> EvalReport:
    results = [sample(value) for value in rr]
    hit_rate = sum(r.hit_rate for r in results) / len(results) if results else 0.0
    return EvalReport(
        k=3,
        mode="dense",
        match_on="chunk_id",
        n_samples=len(results) + failed,
        n_failed=failed,
        metrics={"hit_rate": hit_rate, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
        results=results,
    )


def test_all_zero_without_failures_gets_a_hint() -> None:
    """全 0 且没有失败 —— 这是「标注对不上」的典型指纹，必须提示。"""
    hint = _eval_mismatch_hint(report(rr=[0.0, 0.0, 0.0]))
    assert hint is not None
    assert "切分参数" in hint
    assert "chunk_id" in hint


def test_partial_hits_do_not_get_a_hint() -> None:
    """有命中就不提示 —— 那只是效果差，不是配置错。"""
    assert _eval_mismatch_hint(report(rr=[1.0, 0.0])) is None


def test_failures_do_not_get_a_hint() -> None:
    """有样本失败时走另一条提示路径（打印失败详情），不叠加。"""
    assert _eval_mismatch_hint(report(rr=[], failed=3)) is None


def test_empty_dataset_does_not_get_a_hint() -> None:
    """没有成功样本时不提示「对不上」—— 空结果不该被误诊。"""
    assert _eval_mismatch_hint(report(rr=[])) is None
