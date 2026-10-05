"""Embedding 的公共约定：一个 Protocol，加一个通用的批量驱动器。

这一层只认「能把文本变成向量」这个能力，不认任何具体供应商。
上层（M5 建索引、M6 检索）永远只依赖这个 Protocol。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..config import get_settings
from ..errors import EmbeddingError

__all__ = ["Embedder", "embed_all"]


@runtime_checkable
class Embedder(Protocol):
    """把文本变成向量。

    和 M3 的 Splitter 对比着看

        Splitter:      def split(...)  -> list[Chunk]     ← 纯 CPU，不等待 → 同步
        Embedder: async def embed(...) -> list[list[float]]  ← 要等网络 → 异步

    这里的 async 不是风格选择，是**物理事实**：向量化必须发 HTTP 请求，
    请求期间这个协程什么也做不了，必须把控制权交出去，让别的协程干活。

    这也解释了一个反直觉的现象：纯本地的 HashingEmbedder 明明没有网络，
    为什么也必须写成 async？**因为接口的异步性由契约决定，不由实现决定。**
    调用方要写 ``await embedder.embed(...)``，它不该关心你背后是发网络请求
    还是查字典 —— 那正是 Protocol 存在的意义。
    """

    dim: int
    """向量维度。上层要用它去建 Milvus collection。

    注意：这是协议里的**数据属性**，和 M2 的 ``Parser.extensions`` 一样，
    实现类必须显式标注成 ``int``（不能让它推断成更具体的类型），
    否则会撞上那个「可变属性不变性」的坑。
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本转成向量。

        **返回值的顺序必须和 texts 严格一致** —— 这是硬契约。
        向量和文本一旦错位，检索结果会变得莫名其妙，而且极难排查。
        """
        ...

    async def aclose(self) -> None:
        """释放底层资源（HTTP 连接池等）。没有资源的实现可以是空方法。"""
        ...


async def embed_all(
    embedder: Embedder,
    texts: Sequence[str],
    *,
    batch_size: int | None = None,
    total_timeout: float | None = None,
) -> list[list[float]]:
    """批量向量化：分片 → 并发 → 拍平，全程保持顺序。"""
    if not texts:
        return []

    size = batch_size if batch_size is not None else get_settings().openai_embedding_batch_size
    if size <= 0:
        raise EmbeddingError(f"batch_size 必须为正数，收到 {size}")

    batches = [list(texts[i : i + size]) for i in range(0, len(texts), size)]

    async def run() -> list[list[float]]:
        groups = await asyncio.gather(*(embedder.embed(b) for b in batches))
        return [vec for group in groups for vec in group]

    if total_timeout is None:
        return await run()

    try:
        return await asyncio.wait_for(run(), total_timeout)
    except TimeoutError as exc:
        raise EmbeddingError(
            f"整批向量化超时（{total_timeout} 秒）", source=embedder.__class__.__name__
        ) from exc
