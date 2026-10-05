"""Milvus 实现：chunk 与向量的持久化、检索。

数据库的并发瓶颈在写入锁和磁盘，所以写入闸门比 HTTP 客户端保守得多。
collection 的懒创建用双检锁，避免并发导入时重复建表。
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

# 检索时取回的字段。故意不含 embedding —— 每条命中带 1024 个浮点数
# 会让返回体积暴涨，而检索结果里用不到它。
_SEARCH_OUTPUT_FIELDS = [
    CHUNK_ID_FIELD,
    DOC_ID_FIELD,
    TEXT_FIELD,
    CHUNK_INDEX_FIELD,
    "source",  # 动态字段
]


class MilvusIndexer:
    """管理一个 Milvus collection 里的 chunk 数据（读写都管）。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: AsyncMilvusClient | None = None,
    ) -> None:
        """``AsyncMilvusClient`` 的构造是惰性的，不会立刻连网，真连接发生在第一次 RPC。

        所以构造一个 indexer 很便宜，测试里注入假客户端也不会误连真数据库。
        """
        self._s = settings or get_settings()
        self._collection = self._s.milvus_collection
        self._dim = self._s.openai_embedding_dim
        self._batch_size = self._s.milvus_batch_size

        self._owns_client = client is None
        self._client = client or AsyncMilvusClient(
            uri=self._s.milvus_uri,
            token=self._s.milvus_token,
            timeout=self._s.request_timeout,
        )

        self._ensure_lock = asyncio.Lock()
        self._ready = False
        self._sem = asyncio.Semaphore(self._s.max_concurrency)

    async def __aenter__(self) -> MilvusIndexer:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """关掉自己创建的客户端（注入进来的不动）。"""
        if self._owns_client:
            await self._client.close()

    async def ensure_collection(self) -> None:
        """确保 collection 存在，不存在就建（含索引与 load）。

        双检锁：第一次检查走无锁快速路径（绝大多数调用）；第二次在锁内确认，
        因为等锁期间别的协程可能已经建好了。少了第二检会重复建表。

        ``_ready`` 是缓存，前提是中间没人删掉 collection —— 见 :meth:`drop`。
        """
        if self._ready:
            return

        async with self._ensure_lock:
            if self._ready:
                return

            try:
                if not await self._client.has_collection(self._collection):
                    # pymilvus 的 create_collection 传了 index_params 就会
                    # 自动建索引并 load，不用再手动调两次。
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

    async def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """批量写入 chunk 及其向量，返回写入条数。

        ``chunks`` 和 ``vectors`` 必须一一对应 —— 错位会把「A 的文本」和
        「B 的向量」配成一行，检索结果永远是错的且看不出异常，所以长度不等直接报错。
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
                # 动态字段：schema 没声明，靠 enable_dynamic_field 存下来
                "source": chunk.metadata.get("source", ""),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        batches = [rows[i : i + self._batch_size] for i in range(0, len(rows), self._batch_size)]

        async def write(batch: list[dict[str, Any]]) -> None:
            # 闸门包住整次 upsert，不是只包住「发起请求」那一下
            async with self._sem:
                await self._client.upsert(collection_name=self._collection, data=batch)

        try:
            await asyncio.gather(*(write(batch) for batch in batches))
        except MilvusException as exc:
            raise IndexingError(f"写入 Milvus 失败: {exc}", source=self._collection) from exc

        return len(chunks)

    async def flush(self) -> None:
        """把数据落盘，使其立刻可检索。

        Milvus 是最终一致性的：新数据先进「增长中段」，检索默认不保证马上看得见。
        不 flush 的话，写完立刻查会返回 0 条而且**不报任何错**。

        也可以改成读取时用强一致性，但那会让每一次查询都变慢；
        我们选择把代价压在写入侧一次。
        """
        try:
            await self._client.flush(self._collection)
        except MilvusException as exc:
            raise IndexingError(f"flush 失败: {exc}", source=self._collection) from exc

    async def delete_document(self, doc_id: str) -> int:
        """按 doc_id 删掉该文档的所有 chunk，返回删除条数。

        ⚠️ filter 是字符串拼接，有注入风险。这里的 doc_id 来自 ``stable_id``
        （十六进制串），天然安全；若来源变成用户输入，必须先校验格式。
        Milvus 没有参数化查询，只能自己控制输入。
        """
        expr = f'{DOC_ID_FIELD} == "{doc_id}"'
        try:
            result = await self._client.delete(collection_name=self._collection, filter=expr)
        except MilvusException as exc:
            raise IndexingError(f"删除失败: {exc}", source=self._collection) from exc
        return int(result.get("delete_count", 0))

    async def count(self) -> int:
        """记录数（**近似值**，用于健康检查）。

        统计基于已落盘的 segment，刚写入还没 flush 的数据可能不计入。
        想知道「这次写进去多少条」请用 :meth:`upsert_chunks` 的返回值。
        """
        await self.ensure_collection()
        rows = await self._client.query(
            collection_name=self._collection, filter="", output_fields=["count(*)"]
        )
        return int(rows[0]["count(*)"])

    async def drop(self) -> None:
        """删掉整个 collection。"""
        try:
            if await self._client.has_collection(self._collection):
                await self._client.drop_collection(self._collection)
        except MilvusException as exc:
            raise IndexingError(f"删除 collection 失败: {exc}", source=self._collection) from exc
        # 缓存失效：不清掉的话，下次 ensure_collection() 会以为表还在
        self._ready = False

    @staticmethod
    def _to_scored_chunk(hit: dict[str, Any]) -> ScoredChunk:
        """把一条命中转成 ScoredChunk。

        命中字典的形状是 ``{"<主键字段名>": 值, "distance": 分数, "entity": {...}}``。
        主键那个键名是**字段名本身**（这里是 chunk_id），不是固定的 "id"，
        所以统一从 ``entity`` 里取值。
        """
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
        """向量检索，返回按相关度降序的 ScoredChunk。

        ⚠️ **分数方向**：索引用的是 COSINE，**越大越相似**。
        换成 L2 距离的话方向相反，上层的排序与阈值全部要跟着改，而且不会报错。
        """
        if len(query_vector) != self._dim:
            raise RetrievalError(
                f"查询向量是 {len(query_vector)} 维，但库里的向量是 {self._dim} 维",
                source=self._collection,
            )
        if top_k <= 0:
            raise RetrievalError(f"top_k 必须为正数，收到 {top_k}")

        await self.ensure_collection()

        try:
            hits = await self._client.search(
                collection_name=self._collection,
                # Milvus 支持一次查多个向量，所以接口收「一批」，返回值也是
                # List[List[dict]]，真正的命中在 hits[0]。
                data=[list(query_vector)],
                # 必须显式指定：加 BM25 之后 collection 里有**两个**向量字段
                # （embedding 与 sparse），不指定服务端不知道要搜哪个。
                anns_field=VECTOR_FIELD,
                filter=filter_expr,
                limit=top_k,
                output_fields=list(_SEARCH_OUTPUT_FIELDS),
            )
        except MilvusException as exc:
            raise RetrievalError(f"向量检索失败: {exc}", source=self._collection) from exc

        # 服务端返回的是弱类型 dict，转强类型时的失败方式很多
        # （KeyError / TypeError / ValidationError），统一翻译成本库的异常。
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

        需要两个入参是因为两路要的输入不同：稠密那路要向量，
        关键词那路要**原始文本**（服务端用 BM25 函数现场转成稀疏向量）。

        ``candidate_k`` 是每一路各召回多少条参与融合，默认 ``top_k * 4``。
        候选池必须比最终结果大：RRF 只看每一路给出的**排名**，
        两路都只给 5 条的话，「稠密排第 100、关键词排第 1」的文档根本进不了融合 ——
        而它恰恰是混合检索最该捞回来的那种结果。

        ⚠️ 返回的 score 是 **RRF 融合分**（约 0~0.033），不是余弦相似度。
        两者量纲不同、不可比，所以上层在混合模式下不允许设阈值过滤。
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
        # 两路复用同一个过滤条件：在候选阶段就过滤，比取回来再筛省得多
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

        try:
            return [self._to_scored_chunk(hit) for hit in hits[0]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RetrievalError(
                f"Milvus 返回的命中无法解析（字段缺失或为空）: {exc}",
                source=self._collection,
            ) from exc
