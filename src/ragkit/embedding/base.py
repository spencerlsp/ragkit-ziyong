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

    ★ 和 M3 的 Splitter 对比着看 ★

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
    """批量向量化：分片 → 并发 → 拍平，全程保持顺序。

    TODO(你)：按下面五步实现。

        1) 空输入直接返回：
               if not texts:
                   return []

        2) 解析分片大小并校验：
               size = (
                   batch_size
                   if batch_size is not None
                   else get_settings().openai_embedding_batch_size
               )
               if size <= 0:
                   raise EmbeddingError(f"batch_size 必须为正数，收到 {size}")

           注意别写成 ``batch_size or 默认值``：``0`` 是 falsy，
           传 0 会被静默换成默认值，下面那行校验就永远等不到执行。
           只有 ``None`` 才代表「没提供」。

        3) 切片：
               batches = [list(texts[i : i + size]) for i in range(0, len(texts), size)]

        4) 定义一个内层协程，把「并发 + 拍平」装进去：

               async def run() -> list[list[float]]:
                   groups = await asyncio.gather(*(embedder.embed(b) for b in batches))
                   return [vec for group in groups for vec in group]

           三个要点：
             * ``*(...)`` 是把生成器解包成位置参数。``gather`` 收的是
               「一堆协程」，不是「一个协程列表」—— 写 ``gather(coros)``
               会把列表当成单个协程，直接报错。
             * ``gather`` 和 TaskGroup 的关键差别：**gather 保证结果顺序
               和传入顺序一致**。所以第 1 片的结果一定在 groups[0]，
               拍平之后顺序天然对齐输入。M2 里 TaskGroup 要手动回填下标，
               这里不用 —— 因为 gather 帮我们做了。
               （代价是 gather 的异常语义比较弱：它不会自动取消兄弟任务。
                在「一批失败就整批放弃」的场景里 TaskGroup 更合适。）
             * 双重列表推导 ``[vec for group in groups for vec in group]``
               是按顺序拍平二维列表的标准写法，等价于嵌套两层 for。

        5) 整批超时（可选）：

               if total_timeout is None:
                   return await run()
               try:
                   return await asyncio.wait_for(run(), timeout=total_timeout)
               except TimeoutError as exc:
                   raise EmbeddingError(
                       f"整批向量化超时（{total_timeout} 秒）", source=embedder.__class__.__name__
                   ) from exc

           要点：
             * Python 3.11 起 ``asyncio.TimeoutError`` 就是内置的 ``TimeoutError``，
               所以直接 except TimeoutError 即可。
             * ``wait_for`` 超时会**取消**里面那个协程：正在等待的 HTTP 请求会被
               取消掉，不会再白白跑完。这正是它比「自己 sleep 计时」强的地方。
             * 这里做的是「整批的截止时间」，和 httpx 那个「单次请求超时」是两个
               层次的东西。两者不冲突：单次请求可以在 60 秒内正常完成，
               但整批只给你 30 秒，那就必须整批放弃。

    为什么分片和并发要分成两层：
        服务端对「单次请求里能放多少条文本」有上限（所以必须分片），
        而对「同时能有多少个请求在飞」也有上限（所以在客户端里用 Semaphore 限流）。
        前者是为了让**单个请求合法**，后者是为了让**整体不把服务端打爆**。
    """
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
