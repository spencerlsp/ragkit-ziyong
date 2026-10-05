"""重排（rerank）：用更贵的模型对候选做精排。

**为什么需要重排** —— 这是向量检索的固有短板：
    向量检索为了快，把整段文本压成一个固定长度的向量，这个过程**有损**。
    细节、数字、专有名词都可能在压缩中失真，所以它擅长「大致相关」，
    不擅长「精确命中」。

    重排模型（cross-encoder）反过来：它把 query 和每一篇文档**拼在一起**
    过一遍模型，能同时看见两者的字面交集与细微语义差异，精度高得多。
    代价是慢 —— 它必须对每一对 (query, doc) 单独算一次，
    没法像向量那样预先建好索引。

    所以标准打法是**两阶段**：
        第一阶段（召回）：向量 / 混合检索，快速从几万条里挑出 20~50 条候选；
        第二阶段（精排）：重排模型对这几十条重新打分，取前 k 条。
    **召回保速度，精排保精度。**

一个常见的误解要提前说清楚：如果你只把向量检索的 top-3 拿去重排，
那重排**什么也救不回来** —— 它只能在已有的 3 条里换顺序。
必须先把候选池放大（比如 20 条），重排才有发挥空间。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from ..config import Settings, get_settings
from ..errors import RetrievalError
from ..http import JsonClient

__all__ = ["Reranker", "HttpReranker", "create_reranker"]


@runtime_checkable
class Reranker(Protocol):
    """给「一批候选文本」重新打分的协议。

    注意返回值的形状：``list[tuple[int, float]]``，也就是
    **(原始下标, 分数)** 的列表。为什么不直接返回排好序的文本？
    因为我们手上已经有原文和别的元数据了，再传一遍纯属浪费。
    用下标回指，调用方自己去取 —— 这和 M2 里 TaskGroup 用下标回填结果
    是同一个思路：**传指针比传数据便宜，而且不会因为复制而失同步。**
    """

    async def rerank(
        self, query: str, texts: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """按 query 对 texts 重排，返回 [(原始下标, 分数), ...]，分数降序。"""
        ...

    async def aclose(self) -> None:
        """释放资源。没有资源的实现可以是空方法。"""
        ...


class HttpReranker:
    """调用 ``POST {base_url}/rerank`` 的重排客户端。

    ⚠️ 这**不是** OpenAI 兼容协议 —— OpenAI 至今没有 rerank 端点。
    硅基流动、Cohere、Jina 的 /rerank 接口形状大体相同
    （query + documents 进，results[{index, relevance_score}] 出），
    但字段名和细节有出入。**我们只对硅基流动实测过**，
    换供应商时请先拿一次真实响应核对结构，别照抄。

    关于分数：实测 ``BAAI/bge-reranker-v2-m3`` 返回的是 [0,1] 区间的
    相关度分，且区分度很高 —— 同一个查询下，真正匹配的文档 ≈ 0.995，
    不相关的低到 0.0003 和 0.00002。所以它的分数**可以**用来设阈值，
    但阈值该设多少完全取决于模型，别把某个模型的阈值套到另一个模型上。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self._model = self._s.rerank_model
        self._http = JsonClient(
            base_url=self._s.rerank_base_url,
            headers={"Authorization": "Bearer " + self._s.rerank_api_key.get_secret_value()},
            timeout=self._s.request_timeout,
            max_retries=self._s.max_retries,
            retry_base_delay=self._s.retry_base_delay,
            max_concurrency=self._s.max_concurrency,
            # 重排失败属于「检索过程失败」，所以翻译成 RetrievalError，
            # 而不是 EmbeddingError —— 调用方按业务语义捕获才合理。
            error_cls=RetrievalError,
            source=self._model,
            client=client,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def rerank(
        self, query: str, texts: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """按 query 给 texts 重排。返回 [(原始下标, 分数), ...]，分数降序。"""
        if not texts:
            return []

        payload: dict[str, object] = {
            "model": self._model,
            "query": query,
            "documents": list(texts),
            # 我们手上已经有原文了，让服务端再把文档回传一遍纯属浪费带宽。
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        body = await self._http.post("/rerank", payload)

        results = body.get("results")
        if not isinstance(results, list):
            raise RetrievalError(
                f"重排接口没有返回 results 列表（拿到的是 {type(results).__name__}）",
                source=self._model,
            )

        ranked: list[tuple[int, float]] = []
        for item in results:
            index = int(item["index"])
            # 下标越界必须拦住：一旦错位，你拿到的就是「别处的文本」，
            # 而且表面上完全正常。宁可当场炸。
            if not 0 <= index < len(texts):
                raise RetrievalError(
                    f"重排返回的下标 {index} 越界（共 {len(texts)} 条文本）",
                    source=self._model,
                )
            ranked.append((index, float(item["relevance_score"])))

        # 实测服务端已经按分数降序返回，这里**仍然再排一次**。
        # 理由：排序是一次 O(n log n) 的廉价操作，而「假设服务端排好了」
        # 一旦不成立，后果是「重排之后结果反而更差」—— 这种 bug 极难发现。
        # 廉价的保险，值得买。
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked


def create_reranker(settings: Settings | None = None) -> Reranker:
    """按配置造一个重排器。和 create_embedder 一样，不做全局缓存。"""
    return HttpReranker(settings)
