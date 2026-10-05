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
from .base import VectorStore
from .milvus import MilvusIndexer

__all__ = ["ingest_chunks"]


async def ingest_chunks(
    chunks: Sequence[Chunk],
    *,
    embedder: Embedder | None = None,
    indexer: VectorStore | None = None,
    batch_size: int | None = None,
    replace_documents: bool = True,
    flush: bool = True,
) -> int:
    """把 chunk 向量化并写进 Milvus，返回写入条数。"""
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
