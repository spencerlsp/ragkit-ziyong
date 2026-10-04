"""Milvus 索引层：把 chunk 加向量写进数据库。

这一层的异步特点和 M4 不同：
    M4 的瓶颈在网络往返（可以开 8 个并发）；
    数据库的瓶颈在写入锁和磁盘，并发要**保守**得多（4 个就差不多）。
    同样是 Semaphore，参数背后的道理完全不同。

另一个新东西是 ``asyncio.Lock``：collection 的懒创建必须用双检锁做，
否则并发导入多个文件时会同时建表、互相打架。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pymilvus import AsyncMilvusClient, MilvusException

from ..config import Settings, get_settings
from ..errors import IndexingError
from ..schemas import Chunk
from .schema import DOC_ID_FIELD, build_index_params, build_schema

__all__ = ["MilvusIndexer"]


class MilvusIndexer:
    """管理一个 Milvus collection 里的 chunk 数据。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: AsyncMilvusClient | None = None,
    ) -> None:
        """TODO(你)：五件事。

        1) 解析配置：``self._s = settings or get_settings()``

        2) 存常用值：

               self._collection = self._s.milvus_collection
               self._dim = self._s.openai_embedding_dim
               self._batch_size = self._s.milvus_batch_size

        3) 建客户端（或者用注入进来的）：

               self._owns_client = client is None
               self._client = client or AsyncMilvusClient(
                   uri=self._s.milvus_uri,
                   token=self._s.milvus_token,
                   timeout=self._s.request_timeout,
               )

           和 M4 一样的「谁创建，谁负责销毁」。
           AsyncMilvusClient 的构造是**惰性**的：它不会在这里就连网，
           真正的连接发生在第一次 RPC 时。所以构造一个 indexer 很便宜，
           测试里注入假客户端也不会误连真数据库。

        4) 建「懒创建 collection」需要的两样东西：

               self._ensure_lock = asyncio.Lock()
               self._ready = False

           ``_ready`` 是缓存：一旦确认 collection 存在，后续调用直接返回。
           缓存就要考虑**失效** —— 见 drop() 里的处理。

        5) 建写入并发闸门：

               self._sem = asyncio.Semaphore(self._s.max_concurrency)

           注意这里复用的是同一个配置项，但**含义不同**：
           M4 用它限制「同时在飞的 HTTP 请求数」，
           这里限制「同时在写的数据库批次」。数据库要保守得多。
        """
        # 1) 解析配置
        self._s = settings or get_settings()

        # 2) 缓存常用配置字段
        self._collection = self._s.milvus_collection
        self._dim = self._s.openai_embedding_dim
        self._batch_size = self._s.milvus_batch_size

        # 3) 客户端注入 / 自建，标记是否拥有客户端（用于close）
        self._owns_client = client is None
        self._client = client or AsyncMilvusClient(
            uri=self._s.milvus_uri,
            token=self._s.milvus_token,
            timeout=self._s.request_timeout,
        )

        # 4) 懒创建集合：锁 + ready标记
        self._ensure_lock = asyncio.Lock()
        self._ready = False

        # 5) 写入并发信号量闸门
        self._sem = asyncio.Semaphore(self._s.max_concurrency)

    async def __aenter__(self) -> MilvusIndexer:
        """支持 ``async with MilvusIndexer() as indexer:``。已经写好了。"""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """退出 async with 时自动关掉客户端。已经写好了。"""
        await self.aclose()

    async def aclose(self) -> None:
        """TODO(你)：两行，和 M4 的 embedder 一模一样。

            if self._owns_client:
                await self._client.close()

        注意 pymilvus 3.0 的 AsyncMilvusClient 关闭方法是 **close()**，
        不是 aclose()。这就是为什么我坚持读一遍真实 API 再给你代码 ——
        凭记忆写的话，你会在最后一步拿到 ``AttributeError: aclose``。
        """
        if self._owns_client:
            await self._client.close()

    async def ensure_collection(self) -> None:
        """确保 collection 已存在（不存在就建，并自动建索引 + load）。

        TODO(你)：**双检锁**。这是 M5 最重要的一段代码。

            if self._ready:                    # 第一检：快速路径
                return

            async with self._ensure_lock:
                if self._ready:                # 第二检：等锁期间可能有人建好了
                    return
                try:
                    if not await self._client.has_collection(self._collection):
                        await self._client.create_collection(
                            collection_name=self._collection,
                            schema=build_schema(self._dim),
                            index_params=build_index_params(),
                        )
                except MilvusException as exc:
                    raise IndexingError(
                        f"创建 collection 失败: {exc}", source=self._collection
                    ) from exc
                self._ready = True

        为什么两次检查都不能省：

          * **第一检是性能**：绝大多数调用走到这里就返回了，不用去抢锁排队。
            如果只有第二检，每次写数据都要先排一次队。
          * **第二检是正确性**：你抢到锁的那一刻，世界可能已经变了 ——
            排在你前面的那个任务已经把 collection 建好了。
            少了第二检，你会重复建表，然后收到 "collection already exists"。

        这个模式不只属于 asyncio：单例、缓存预热、连接池初始化全是这个形状。

        顺带一个 pymilvus 的便利设计：
        ``create_collection(schema=..., index_params=...)`` **会自动建索引并 load**
        （见源码里的 ``_create_collection_with_schema``）。
        所以一次调用就搞定了「建表 + 建索引 + 加载」，不用手动调两次。

        ``self._ready`` 这个缓存有个前提：**中间没人把 collection 删掉**。
        外部手动 drop 的话缓存就失效了 —— 这是缓存的固有代价。
        我们自己提供的 drop() 会负责清掉它。
        """
        # 第一检：快速路径，无锁，绝大多数请求直接返回
        if self._ready:
            return

        # 进入异步锁，保证同一时刻只有一个协程进入初始化逻辑
        async with self._ensure_lock:
            # 第二检：等待锁的过程中，别的协程可能已经完成创建
            if self._ready:
                return

            try:
                # 判断集合是否存在
                if not await self._client.has_collection(self._collection):
                    await self._client.create_collection(
                        collection_name=self._collection,
                        schema=build_schema(self._dim),
                        index_params=build_index_params(),
                    )
            except MilvusException as exc:
                raise IndexingError(
                    f"创建 collection 失败: {exc}", source=self._collection
                ) from exc
            # 标记集合已经就绪，后续调用走快速路径
            self._ready = True

    async def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """把 chunk 和它对应的向量一起写进 Milvus，返回写入条数。

        TODO(你)：五步。

        1) 校验长度：

               if len(chunks) != len(vectors):
                   raise IndexingError(
                       f"chunk 数（{len(chunks)}）和向量数（{len(vectors)}）不一致",
                       source=self._collection,
                   )
               if not chunks:
                   return 0

           为什么必须校验：调用方多半是分两步拿到这两个序列的
           （先 embed，再传进来）。中间任何一步错位，都会把
           「A 的文本」和「B 的向量」配成一行写进库 —— 之后检索出来的
           内容永远是错的，而且完全看不出异常。**宁可当场炸。**

        2) ``await self.ensure_collection()``

        3) 组装行（每行一个 dict，键名必须和 schema 里的字段名**完全一致**）：

               rows: list[dict[str, Any]] = [
                   {
                       "chunk_id": chunk.chunk_id,
                       "doc_id": chunk.doc_id,
                       "text": chunk.text,
                       "chunk_index": chunk.index,
                       "embedding": list(vector),
                       "source": chunk.metadata.get("source", ""),
                   }
                   for chunk, vector in zip(chunks, vectors, strict=True)
               ]

           最后那个 ``source`` 是**动态字段**（schema 里没声明，靠
           ``enable_dynamic_field=True`` 存下来）。

           ``zip(..., strict=True)``（Python 3.10+）会在两边长度不等时直接抛错。
           第 1 步已经查过了 —— 那为什么还要 strict？
           因为这是**免费的二次保险**：将来有人重构时删掉了第 1 步的校验，
           这里还能兜住。而且它顺带告诉读代码的人「这两条序列必须一一对应」。
           （ruff 的 B905 规则也会强制你写 strict 参数，不写就报警。）

        4) 分批：

               batches = [
                   rows[i : i + self._batch_size]
                   for i in range(0, len(rows), self._batch_size)
               ]

        5) 并发写入 + 异常翻译：

               async def write(batch: list[dict[str, Any]]) -> None:
                   async with self._sem:
                       await self._client.upsert(
                           collection_name=self._collection, data=batch
                       )

               try:
                   await asyncio.gather(*(write(batch) for batch in batches))
               except MilvusException as exc:
                   raise IndexingError(
                       f"写入 Milvus 失败: {exc}", source=self._collection
                   ) from exc

               return len(chunks)

           闸门 ``async with self._sem`` 的位置又是关键：
           必须包住**整个 upsert 调用**，不是只包住组装 batch 那一下。
        """
        if len(chunks) != len(vectors):
            raise IndexingError(
                f"chunk 数（{len(chunks)}）和向量数（{len(vectors)}）不一致",
                source=self._collection,
            )
        if not chunks:
            return 0

        await self.ensure_collection()

        rows: list[dict[str, Any]] = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "chunk_index": chunk.index,
                "embedding": list(vector),
                "source": chunk.metadata.get("source", ""),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        batches = [rows[i : i + self._batch_size] for i in range(0, len(rows), self._batch_size)]

        async def write(batch: list[dict[str, Any]]) -> None:
            async with self._sem:
                await self._client.upsert(
                    collection_name=self._collection,
                    data=batch,
                )

        try:
            await asyncio.gather(*(write(batch) for batch in batches))
        except MilvusException as exc:
            raise IndexingError(f"写入 Milvus 失败: {exc}", source=self._collection) from exc

        return len(chunks)

    async def delete_document(self, doc_id: str) -> int:
        """按 doc_id 删掉这个文档的所有 chunk，返回删除条数。

        TODO(你)：

            expr = f'{DOC_ID_FIELD} == "{doc_id}"'
            try:
                result = await self._client.delete(
                    collection_name=self._collection, filter=expr
                )
            except MilvusException as exc:
                raise IndexingError(f"删除失败: {exc}", source=self._collection) from exc
            return int(result.get("delete_count", 0))

        两个要点：

          * **拼接 filter 字符串是有注入风险的。** 如果 doc_id 里带一个引号，
            整个表达式就被破坏了（甚至可能变成删除全部数据）。我们的 doc_id
            是 stable_id 生成的十六进制串，天然安全；但只要哪天它变成用户输入，
            就必须先校验格式。Milvus 没有参数化查询，只能靠你自己控制输入。
          * ``delete`` 返回的是 ``{"delete_count": N}``。用 ``.get(..., 0)`` 取值，
            别用 ``[]`` —— 少一个键不该让你的程序 KeyError。
        """
        expr = f'{DOC_ID_FIELD} == "{doc_id}"'
        try:
            result = await self._client.delete(collection_name=self._collection, filter=expr)
        except MilvusException as exc:
            raise IndexingError(f"删除失败: {exc}", source=self._collection) from exc
        return int(result.get("delete_count", 0))

    async def count(self) -> int:
        """返回 collection 里的记录数（只适合做健康检查）。

        TODO(你)：三步。

            1) ``await self.ensure_collection()``
            2) rows = await self._client.query(
                   collection_name=self._collection,
                   filter="",
                   output_fields=["count(*)"],
               )
            3) ``return int(rows[0]["count(*)"])``

        ⚠️ 这个数字是**近似**的：Milvus 的统计基于已经落盘的 segment，
        刚写进去、还没 flush 的数据可能不计入。
        拿它做健康检查可以，别拿它当「我到底写进去了多少条」的依据 ——
        那个数应该由 upsert 的返回值给你（所以 upsert 才会 return 条数）。
        """
        await self.ensure_collection()
        rows = await self._client.query(
            collection_name=self._collection, filter="", output_fields=["count(*)"]
        )
        return int(rows[0]["count(*)"])

    async def drop(self) -> None:
        """删掉整个 collection。

        TODO(你)：三步，注意最后一步。

            1) ``if await self._client.has_collection(self._collection):``
            2) ``await self._client.drop_collection(self._collection)``
            3) ``self._ready = False``

        用 try/except MilvusException 包住前两步，翻译成 IndexingError。

        为什么要重置 ``_ready``：它是**缓存**。collection 都没了，
        如果不重置，下一次 ``ensure_collection()`` 会以为它还在，
        于是跳过创建，然后所有写入都报「collection not found」。

        **凡是缓存，就一定要问「它在什么情况下会失效」**，
        然后在那个地方主动清掉。
        """
        try:
            if await self._client.has_collection(self._collection):
                await self._client.drop_collection(self._collection)
        except MilvusException as exc:
            raise IndexingError(f"删除 collection 失败: {exc}", source=self._collection) from exc
        self._ready = False
