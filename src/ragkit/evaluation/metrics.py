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
    """前 k 条里**有没有**相关内容（命中就是 1.0，没命中就是 0.0）。

    TODO(你)：一行。

        返回 1.0 当 retrieved[:k] 里存在任何一个属于 relevant 的元素，否则 0.0。
        提示：``any(rid in relevant for rid in retrieved[:k])``，
              记得把布尔值转成浮点（``1.0 if ... else 0.0``）。

    这是最宽松也最常用的指标：只关心「捞到了没有」，不关心捞到几条、排第几。
    适合做第一道体检 —— 如果连 HitRate 都很低，后面几个指标没必要看。
    """
    return 1.0 if any(rid in relevant for rid in retrieved[:k]) else 0.0
    # raise NotImplementedError("TODO: 一行，见上面的提示")


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """前 k 条覆盖了**多少比例**的正确答案。

    TODO(你)：三步。

        1) ``if not relevant: return 0.0``
           —— 没有标准答案的样本，召回率是**无定义**的。
              这里返回 0.0 是一种约定（也可以选择跳过这条样本）。
              关键是**明确一种约定并写进 docstring**，
              而不是让它变成除零异常或者静默的 NaN。

        2) 统计命中的**去重**个数：
               hits = len(set(retrieved[:k]) & set(relevant))

           为什么要去重：万一检索结果里有重复 ID（数据没清理干净、
           或者我们拼错了两路结果），重复项会被算两次，把召回率虚高。
               ``&`` 是集合交集运算。

        3) ``return hits / len(relevant)``

    它和 hit_rate 的区别：hit_rate 问「有没有捞到」，recall 问「捞全了没有」。
    如果标准答案有 3 个相关内容，你只捞到 1 个：
    hit_rate = 1.0，但 recall = 0.33 —— 两个都要看。
    """
    if not relevant:
        return 0.0
    # 注意 set(...) 只包住切片，交集的另一边已经是集合了。
    # 写成 set(a & b) 的话，a 还是 list，`list & set` 会直接 TypeError。
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """第一个相关结果的排名倒数（第 1 名得 1.0，第 2 名得 0.5，第 3 名得 0.33…）。

    TODO(你)：一个循环。

            for rank, rid in enumerate(retrieved[:k], start=1):
                if rid in relevant:
                    return 1.0 / rank
            return 0.0

    两个细节：
      * ``enumerate(..., start=1)`` —— **排名从 1 开始**。
        如果从 0 开始，第 1 名会算出 1/0 除零错误。
        这个 off-by-one 是检索指标里最经典的坑。
      * 找到第一个就 ``return`` —— 后面的排名我们不关心。
        RR 只衡量**第一个**相关结果的位置。

    为什么要「倒数」而不是直接用排名：这样分数落在 0~1 之间，
    而且「第 1 名」和「第 2 名」的差距（1.0 vs 0.5）明显大于
    「第 9 名」和「第 10 名」的差距（0.11 vs 0.10）——
    符合直觉：把相关内容从第 10 名提到第 9 名，价值远小于从第 2 名提到第 1 名。

    多条样本上取平均，就是常说的 **MRR（Mean Reciprocal Rank）**。
    """
    for rank, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """NDCG@k：带位置折扣的排序质量分（1.0 = 完美排序）。

    TODO(你)：三步。

        NDCG 的思路：越靠前的命中越值钱，但「值钱」是**对数衰减**的。
        第 i 位的权重是 ``1 / log2(i + 1)``：
            第 1 位 1.0、第 2 位 0.63、第 3 位 0.5、第 4 位 0.43……

        1) 算 DCG（实际排序的得分）：

               dcg = 0.0
               for rank, rid in enumerate(retrieved[:k], start=1):
                   if rid in relevant:
                       dcg += 1.0 / math.log2(rank + 1)

           （这里用的是「二元相关性」：相关就是 1，不相关就是 0。
             生产里还有分级相关性——「高度相关 / 部分相关」，
             但那就需要人工标注分级，成本高得多。）

        2) 算 IDCG（**理想排序**的得分）：

               ideal_hits = min(len(relevant), k)
               idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

           IDCG 就是「假设所有相关内容都排在最前面」时的得分，
           它同时也是归一化因子，把结果压到 0~1。

           ``min(len(relevant), k)`` 这个上界不能省：
           如果标准答案有 10 个，但你只看前 3 条，
           那「理想情况」也只能命中 3 个。用 10 去算 IDCG 会让分母虚大，
           得分永远上不去。

        3) 当心分母为零：

               return dcg / idcg if idcg > 0 else 0.0

           IDCG 为 0 意味着 relevant 是空的，此时 NDCG 无定义（同 recall）。

    为什么 NDCG 比 HitRate / Recall 更有信息量：
    它同时考虑了「捞到几个」和「排得怎么样」。
    同样捞到 2 条相关内容，一条排在第 1、2 位，另一条排在第 9、10 位，
    HitRate 和 Recall 给出的分数完全一样，NDCG 能区分出来。
    """

    dcg = 0.0
    for rank, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0
