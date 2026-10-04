"""评估层：用数据回答「检索效果到底怎么样」。

对外入口：
    evaluate()        跑一遍评估集，返回 EvalReport
    load_dataset()    从 JSONL 读评估集
    dump_dataset()    写回 JSONL
    EvalSample        一条样本（问题 + 正确 ID）
    四个指标函数       纯函数，可以单独拿来用

典型用法（对比稠密 vs 混合）：

    samples = load_dataset("data/eval.jsonl")
    async with Retriever(settings=settings) as retriever:
        dense = await evaluate(samples, retriever)
        hybrid = await evaluate(samples, retriever, mode="hybrid")
    print(dense.summary())
    print(hybrid.summary())

**这一层存在的意义**：让「换个切分参数」「换个 embedding 模型」
「加个混合检索」这类决定，从「感觉好像好一点」变成「NDCG 从 0.61 涨到 0.74」。
"""

from __future__ import annotations

from .dataset import EvalSample, dump_dataset, load_dataset
from .metrics import hit_rate, ndcg_at_k, recall_at_k, reciprocal_rank
from .runner import EvalReport, FailureRecord, SampleResult, evaluate

__all__ = [
    "EvalSample",
    "load_dataset",
    "dump_dataset",
    "evaluate",
    "EvalReport",
    "SampleResult",
    "FailureRecord",
    "hit_rate",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
]
