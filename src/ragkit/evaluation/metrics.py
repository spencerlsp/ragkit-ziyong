"""检索评估的四个经典指标。

全是**纯函数**：输入一串有序的「检索结果 ID」和一个「正确答案集合」，
输出一个 0~1 的分数。不碰网络、不碰数据库、不用 LLM。

为什么这点很重要：**评估指标本身必须绝对可信。**
如果连算分都要怀疑，你永远分不清「混合检索真的更好」和「我的指标算错了」。
所以这四个函数是全项目里最该测试、也最该用确定性数值钉死的部分。

约定：
    * ``retrieved`` 是**按相关度降序**排列的 ID 列表（第 0 个最相关）；
    * ``relevant`` 是正确答案的 ID 集合（顺序无关）；
    * ``k`` 是只看前 k 条。
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

__all__ = ["hit_rate", "recall_at_k", "reciprocal_rank", "ndcg_at_k"]


def hit_rate(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """前 k 条里**有没有**相关内容（命中就是 1.0，没命中就是 0.0）。"""
    return 1.0 if any(rid in relevant for rid in retrieved[:k]) else 0.0
    # raise NotImplementedError("TODO: 一行，见上面的提示")


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """前 k 条覆盖了**多少比例**的正确答案。"""
    if not relevant:
        return 0.0
    # 注意 set(...) 只包住切片，交集的另一边已经是集合了。
    # 写成 set(a & b) 的话，a 还是 list，`list & set` 会直接 TypeError。
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """第一个相关结果的排名倒数（第 1 名得 1.0，第 2 名得 0.5，第 3 名得 0.33…）。"""
    for rank, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """NDCG@k：带位置折扣的排序质量分（1.0 = 完美排序）。"""

    dcg = 0.0
    for rank, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0
