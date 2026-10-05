"""高层入口：把一批 chunk 完整地送进 Milvus。

顺序很重要，记住这个三段式：

    delete(doc_id)  ->  embed(texts)  ->  upsert(rows)

**先删后写**，这样重复导入同一份文件不会累积重复数据，
改了切分参数也不会在库里留下孤儿。
反过来的话（先写后删）会把刚写进去的新数据一起删掉。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..embedding import Embedder, create_embedder, embed_all
from ..schemas import Chunk
from .milvus import MilvusIndexer

__all__ = ["ingest_chunks"]


async def ingest_chunks(
    chunks: Sequence[Chunk],
    *,
    embedder: Embedder | None = None,
    indexer: MilvusIndexer | None = None,
    batch_size: int | None = None,
    replace_documents: bool = True,
    flush: bool = True,
) -> int:
    """把 chunk 向量化并写进 Milvus，返回写入条数。

    TODO(你)：按下面的骨架实现。

        if not chunks:
            return 0

        # 「谁创建，谁负责销毁」—— M4 学到的原则，在组合两个组件时更关键
        owns_embedder = embedder is None
        owns_indexer = indexer is None
        embedder = embedder or create_embedder()
        indexer = indexer or MilvusIndexer()

        try:
            await indexer.ensure_collection()

            if replace_documents:
                for doc_id in {chunk.doc_id for chunk in chunks}:
                    await indexer.delete_document(doc_id)

            vectors = await embed_all(
                embedder, [chunk.text for chunk in chunks], batch_size=batch_size
            )
            return await indexer.upsert_chunks(chunks, vectors)
        finally:
            if owns_embedder:
                await embedder.aclose()
            if owns_indexer:
                await indexer.aclose()

    四个要点：

      1) ``{chunk.doc_id for chunk in chunks}`` 是个**集合推导**，顺手去重。
         一批 chunk 通常来自同一个文档，但接口允许混合来源；
         对同一个 doc_id 删两次是浪费（还可能踩到删除的并发坑）。

      2) **删除是串行的，不走并发**。只是几次 RPC，串行足够快；
         而且并发删除同一个 collection 更容易出问题。
         这是个刻意的取舍：**并发不是免费的，只在收益明显时才用。**

      3) **``finally`` 里只关自己创建的东西**。如果调用方传了 embedder 进来，
         他可能还要继续用，你关掉就是替他做了决定。
         这是 M4 那条「谁创建，谁负责销毁」在**组合两个组件**时的直接应用 ——
         组合场景比单组件更容易犯这个错。

      4) **为什么先 ensure_collection 再删除**：
         delete_document 作用在一个不存在的 collection 上会报错。
         先确保它存在，删除一个空 collection 是安全的（删 0 条）。

    返回 upsert 的条数，也就是真正写进去的 chunk 数。
    """
    if not chunks:
        return 0

    # 标记所有权：判断是否是本函数新建的embedder/indexer
    owns_embedder = embedder is None
    owns_indexer = indexer is None

    # 没有传入就新建实例
    embedder = embedder or create_embedder()
    indexer = indexer or MilvusIndexer()

    try:
        # 先保证collection存在，否则delete会报错
        await indexer.ensure_collection()

        # 开启文档覆盖模式：先删除这批chunks对应的所有旧文档
        if replace_documents:
            # 集合推导，自动去重doc_id，避免重复删除同一个文档
            unique_doc_ids = {chunk.doc_id for chunk in chunks}
            for doc_id in unique_doc_ids:
                await indexer.delete_document(doc_id)

        # 提取所有chunk文本，批量向量化
        texts = [chunk.text for chunk in chunks]
        vectors = await embed_all(embedder, texts, batch_size=batch_size)

        written = await indexer.upsert_chunks(chunks, vectors)

        # 写入后 flush 一次，保证「写完立刻能查到」。
        #
        # 不加这一步，紧接着的检索很可能返回 0 条，而且**不报错** ——
        # Milvus 是最终一致性的，新数据要先落盘才保证可见。
        # （代价是写入变慢一点。持续大批量导入时应该关掉它，
        #   等全部写完之后统一 flush 一次，否则会产出大量小 segment，反而拖慢查询。）
        if flush:
            await indexer.flush()
        return written

    finally:
        # 资源释放：只关闭本函数自己创建的实例，外部传入的交给调用方管理
        if owns_embedder:
            await embedder.aclose()
        if owns_indexer:
            await indexer.aclose()
