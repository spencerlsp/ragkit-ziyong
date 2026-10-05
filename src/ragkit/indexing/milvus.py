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

from pymilvus import AnnSearchRequest, AsyncMilvusClient, MilvusException, RRFRanker

from ..config import Settings, get_settings
from ..errors import IndexingError, RetrievalError
from ..schemas import Chunk, ScoredChunk
from .schema import (
    CHUNK_ID_FIELD,
    CHUNK_INDEX_FIELD,
    DOC_ID_FIELD,
    SPARSE_FIELD,
    TEXT_FIELD,
    VECTOR_FIELD,
    build_index_params,
    build_schema,
)

__all__ = ["MilvusIndexer"]

# 检索时要取回的字段。
# ⚠️ 故意**不包含 embedding** —— 每条命中带 1024 个浮点数会让返回体积暴涨几十倍，
#    而检索结果里我们根本用不到它（真要向量的话重新 embed 一次更省事）。
_SEARCH_OUTPUT_FIELDS = [
    CHUNK_ID_FIELD,
    DOC_ID_FIELD,
    TEXT_FIELD,
    CHUNK_INDEX_FIELD,
    "source",  # 动态字段
]


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

    async def flush(self) -> None:
        """把内存里的数据落盘，让它们**立刻**能被检索到。

        为什么需要它 —— 这是端到端实跑才暴露出来的问题：
        ``upsert`` 报告写了 4 条，紧接着 ``search`` 返回 0 条，而且**不报任何错**。

        原因是 Milvus 的**最终一致性**：新写入的数据先进「增长中段」
        (growing segment)，检索默认不保证立刻看得见它。
        第一次跑 demo 时碰巧赶上了（查到了 2 条），第二次就没赶上 ——
        这种「时好时坏」比稳定失败更难查。

        要立刻可见有两条路：
          * 读的时候指定强一致性（``consistency_level="Strong"``），
            每次查询都要等服务端确认最新状态，延迟明显变高；
          * 写完之后 ``flush`` 一次，把增长段封存并建索引。

        我们选后者：**代价集中在写入侧一次，而不是让每一次查询都变慢。**
        这也符合 RAG 的访问模式 —— 写入是批量的、偶发的，查询是频繁的。
        """
        try:
            await self._client.flush(self._collection)
        except MilvusException as exc:
            raise IndexingError(f"flush 失败: {exc}", source=self._collection) from exc

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

    @staticmethod
    def _to_scored_chunk(hit: dict[str, Any]) -> ScoredChunk:
        """把 Milvus 的一条命中转成 ScoredChunk。已经写好了。

        命中字典的形状是 ``{"<主键字段名>": 值, "distance": 分数, "entity": {...}}``。
        注意主键那个键名是**字段名本身**（我们这里是 "chunk_id"），
        不是固定的 "id" —— 官方文档的例子里主键正好叫 id，容易看错。
        所以最稳的读法是从 ``entity`` 里取（output_fields 里要了它）。
        """
        # 用 `or {}` 而不是默认参数：服务端可能返回 entity=None，
        # 那种情况下 get 的默认值不会生效，后面 .get 会直接 AttributeError。
        entity = hit.get("entity") or {}
        return ScoredChunk(
            chunk=Chunk(
                chunk_id=str(entity.get(CHUNK_ID_FIELD, "")),
                doc_id=str(entity.get(DOC_ID_FIELD, "")),
                text=str(entity.get(TEXT_FIELD, "")),
                index=int(entity.get(CHUNK_INDEX_FIELD, 0)),
                metadata={"source": str(entity.get("source", ""))},
            ),
            score=float(hit["distance"]),
        )

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[ScoredChunk]:
        """用查询向量检索，返回按相关度降序的 ScoredChunk。

        TODO(你)：五步。

        1) 校验查询向量的维度：

               if len(query_vector) != self._dim:
                   raise RetrievalError(
                       f"查询向量是 {len(query_vector)} 维，但库里的向量是 {self._dim} 维",
                       source=self._collection,
                   )

           这是第四道维度防线了。为什么这里也要拦？
           因为查询向量是**调用方算出来的**：他可能用了另一个 embedding 模型，
           或者拼接时搞错了。维度不匹配时服务端给的错误很含糊，
           在这里拦住能直接告诉他「768 vs 1024」。

        2) 校验 top_k：

               if top_k <= 0:
                   raise RetrievalError(f"top_k 必须为正数，收到 {top_k}")

        3) ``await self.ensure_collection()``

        4) 调用检索：

               hits = await self._client.search(
                   collection_name=self._collection,
                   data=[list(query_vector)],
                   anns_field=VECTOR_FIELD,
                   filter=filter_expr,
                   limit=top_k,
                   output_fields=list(_SEARCH_OUTPUT_FIELDS),
               )

           三个细节：
             * ``data=[list(query_vector)]`` —— Milvus 支持一次查多个向量，
               所以接口收的是「一批查询向量」。我们只查一个也要包一层。
               返回值同理是 ``List[List[dict]]``，真正的命中在 ``hits[0]``。
             * **必须显式传 ``anns_field=VECTOR_FIELD``。**
               这是加了 BM25 之后才暴露出来的问题：collection 里现在有
               **两个**向量字段（稠密 embedding + 稀疏 sparse），
               不指定的话服务端不知道你要搜哪个，直接报
               ``multiple anns_fields exist, please specify a anns_field``。
               —— 这正是端到端实跑的价值：165 条离线测试一条都发现不了它，
               因为假客户端不校验这个参数。
             * **不传 search_params**。索引是用 COSINE 建的，服务端默认就用它；
               显式传一个和索引不一致的 metric_type 反而会报错。
               这也说明「度量方式」是**建索引那一刻定下的全局约定**，
               不是每次查询的选项。

           别让 MilvusException 漏出去，用 try/except 翻译成 RetrievalError。

        5) 转换 —— 而且是**带异常翻译的转换**：

               try:
                   return [self._to_scored_chunk(hit) for hit in hits[0]]
               except (KeyError, TypeError, ValueError) as exc:
                   raise RetrievalError(
                       f"Milvus 返回的命中无法解析（字段缺失或为空）: {exc}",
                       source=self._collection,
                   ) from exc

           为什么要包一层：服务端返回的是**弱类型的 dict**，
           转成 ScoredChunk 这种强类型模型时，失败方式五花八门 ——
             * 少个字段 → KeyError
             * entity 不是 dict、distance 不是数字 → TypeError
             * text 是空字符串 → pydantic 的 ValidationError（它是 ValueError 的子类）
           不翻译的话，最后一种会漏出一个 pydantic 异常，
           而调用方写的是 ``except RetrievalError`` —— 兜不住。

           **「把弱类型数据转成强类型」是每个外部接口都有的边界，
             边界上的异常必须统一成自己的类型。**

        ⚠️ **分数方向**：COSINE 是**越大越相似**，Milvus 也已经按分数降序返回。
           如果哪天把索引换成 L2（距离，越小越相似），
           整个上层的排序、阈值、乃至「取前 k 条」都要跟着反过来 ——
           而这不会报任何错。度量方式的方向必须一路传导到最上层。
        """
        # 1. 校验向量维度
        if len(query_vector) != self._dim:
            raise RetrievalError(
                f"查询向量是 {len(query_vector)} 维，但库里的向量是 {self._dim} 维",
                source=self._collection,
            )

        # 2. 校验top_k合法性
        if top_k <= 0:
            raise RetrievalError(f"top_k 必须为正数，收到 {top_k}")

        # 3. 确保集合存在
        await self.ensure_collection()

        # 4. Milvus向量检索，捕获底层异常
        try:
            hits = await self._client.search(
                collection_name=self._collection,
                data=[list(query_vector)],
                anns_field=VECTOR_FIELD,
                filter=filter_expr,
                limit=top_k,
                output_fields=list(_SEARCH_OUTPUT_FIELDS),
            )
        except MilvusException as exc:
            raise RetrievalError(f"Milvus向量检索失败: {exc}", source=self._collection) from exc

        # 5. 转换：弱类型 dict -> 强类型 ScoredChunk，失败统一翻译
        try:
            return [self._to_scored_chunk(hit) for hit in hits[0]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RetrievalError(
                f"Milvus 返回的命中无法解析（字段缺失或为空）: {exc}",
                source=self._collection,
            ) from exc

    async def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int = 5,
        filter_expr: str = "",
        candidate_k: int | None = None,
    ) -> list[ScoredChunk]:
        """混合检索：稠密向量 + BM25 关键词，两路结果用 RRF 融合。

        为什么需要两个入参（向量 + 原文）：
            稠密那一路要向量；稀疏那一路要**原始查询文本** ——
            服务端会用 BM25 函数现场把文本转成稀疏向量，所以我们传文本。
            同一个查询的两种表示，缺一不可。

        RRF（Reciprocal Rank Fusion，倒数排名融合）是什么：
            对每一路结果里的第 i 名，贡献 ``1 / (k + i)``（k 默认 60），
            再把两路的贡献相加作为最终分数。

            它最大的好处是**不需要归一化**。余弦相似度是 0~1，
            BM25 分数可能是 0~30，量纲完全不同，直接加权相加毫无意义；
            而 RRF 只看**排名**、不看分数，天然绕开了量纲问题。
            这就是它成为混合检索默认选择的原因。

            代价是：融合后我们**丢掉了原始分数信息**。
            想保留可比的分数，得改用 ``WeightedRanker``（但要先自己把两路
            分数归一化到同一量纲，那才是真正的麻烦）。

        参数 ``candidate_k``：每一路各召回多少条来参与融合。
            默认是 ``top_k * 4``。为什么候选池要比最终结果大：
            RRF 只能看见每一路**给出的排名**。如果两路都只给 5 条，
            那么「稠密排第 100、关键词排第 1」的文档根本进不了融合 ——
            而它恰恰是混合检索最该捞回来的那种结果。
            候选池越大融合越有信息，代价是每路的检索开销。

        ⚠️ **混合模式的 score 不是余弦相似度，是 RRF 分数。**
           量级大约 0~0.033（k=60 时，两路各最多贡献 1/60）。
           和稠密模式的 0~1 完全不可比 ——
           所以 Retriever 在 hybrid 模式下会**拒绝** score_threshold，
           否则你用 0.7 当阈值会把结果全部滤光，还以为是「搜不到」。
        """
        if not query_text.strip():
            raise RetrievalError(
                "混合检索的 query_text 不能为空（BM25 那一路需要原始文本）",
                source=self._collection,
            )
        if len(query_vector) != self._dim:
            raise RetrievalError(
                f"查询向量是 {len(query_vector)} 维，但库里的向量是 {self._dim} 维",
                source=self._collection,
            )
        if top_k <= 0:
            raise RetrievalError(f"top_k 必须为正数，收到 {top_k}")

        await self.ensure_collection()

        pool = candidate_k if candidate_k is not None else top_k * 4
        # 两个分支复用同一个过滤条件：只在候选阶段就过滤，
        # 比「先各取一堆再在客户端过滤」省得多。
        expr = filter_expr or None

        try:
            hits = await self._client.hybrid_search(
                collection_name=self._collection,
                reqs=[
                    AnnSearchRequest(
                        data=[list(query_vector)],
                        anns_field=VECTOR_FIELD,
                        param={"metric_type": "COSINE"},
                        limit=pool,
                        expr=expr,
                    ),
                    AnnSearchRequest(
                        data=[query_text],
                        anns_field=SPARSE_FIELD,
                        param={"metric_type": "BM25"},
                        limit=pool,
                        expr=expr,
                    ),
                ],
                ranker=RRFRanker(),
                limit=top_k,
                output_fields=list(_SEARCH_OUTPUT_FIELDS),
            )
        except MilvusException as exc:
            raise RetrievalError(f"混合检索失败: {exc}", source=self._collection) from exc

        # 转换失败也要翻译 —— 和 dense 的 search() 保持一致
        try:
            return [self._to_scored_chunk(hit) for hit in hits[0]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RetrievalError(
                f"Milvus 返回的命中无法解析（字段缺失或为空）: {exc}",
                source=self._collection,
            ) from exc
