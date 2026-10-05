"""向量库的抽象接口。

为什么要有它：``Ragkit`` 门面、``Retriever``、``ingest_chunks`` 都只关心
「一个能存能查的东西」，不该关心它背后是 Milvus 还是别的。

好处有两个，**第二个更重要**：

  1) 以后接 Qdrant / pgvector，只写一个新的实现类，上层一行不用改；
  2) 测试可以在**完全没有数据库**的情况下跑完整条链路 ——
     不用再造一个几百行的假 Milvus 客户端。
     （这个项目里已经出现过三个手抄的假客户端，然后它们开始漂移，
       害得测试报了个莫名其妙的 AttributeError。抽象成协议就不会再有这种事。）

注意它是 Protocol：``MilvusIndexer`` **什么都不用继承**，形状对得上就算数。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..schemas import Chunk, ScoredChunk

__all__ = ["VectorStore"]


@runtime_checkable
class VectorStore(Protocol):
    """chunk + 向量的持久化与检索接口。

    ★ 和 M4 的 ``Embedder``、M3 的 ``Splitter`` 是同一套路 ★
    定义一个「能力」，让上层依赖能力而不是某个具体数据库。
    """

    async def ensure_collection(self) -> None:
        """确保底层存储已就绪（建表 / 建索引 / 加载）。"""
        ...

    async def upsert_chunks(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        """写入 chunk 及其向量，返回写入条数。"""
        ...

    async def delete_document(self, doc_id: str) -> int:
        """删掉某个文档的所有 chunk，返回删除条数。"""
        ...

    async def flush(self) -> None:
        """把数据落盘，使其立刻可检索（最终一致性的存储才需要）。"""
        ...

    async def count(self) -> int:
        """返回记录数（近似值即可，用于健康检查）。"""
        ...

    async def drop(self) -> None:
        """删掉整张表。"""
        ...

    async def search(
        self, query_vector: Sequence[float], *, top_k: int = 5, filter_expr: str = ""
    ) -> list[ScoredChunk]:
        """向量检索，返回按相关度降序的 ScoredChunk。"""
        ...

    async def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int = 5,
        filter_expr: str = "",
        candidate_k: int | None = None,
    ) -> list[ScoredChunk]:
        """混合检索（稠密 + 关键词），返回按融合分降序的 ScoredChunk。"""
        ...

    async def aclose(self) -> None:
        """释放资源。"""
        ...
