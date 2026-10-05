"""测试替身的正式落点。

**新增测试请从这里取，不要再在测试文件里另抄一份。**

这个项目里已经出现过三份手抄的 ``FakeMilvusClient``，然后它们开始漂移 ——
一个文件有 ``names()``、另一个没有，害得测试报了个莫名其妙的 AttributeError。
手抄的夹具不会报错，只会让下次抄漏显得毫无道理。

现在有了 ``VectorStore`` 协议，连假 Milvus 客户端都不需要了：
``StubStore`` 只有几十行，而且**不依赖 pymilvus**。

（把现存那三份迁移到这里是个独立的清理任务。它们各自的行为细节
  已经有些差异，需要逐个核对 —— 不适合在一次功能开发里顺手做。）
"""

from __future__ import annotations

from collections.abc import Sequence

from ragkit.schemas import Chunk, ScoredChunk

__all__ = ["StubEmbedder", "StubReranker", "StubStore", "make_chunk"]


def make_chunk(text: str, doc_id: str = "d1", index: int = 0) -> Chunk:
    """造一个合法的 Chunk（chunk_id 用内容哈希，和真实切分一致）。"""
    from ragkit.utils import stable_id

    return Chunk(
        chunk_id=stable_id(doc_id, str(index), text),
        doc_id=doc_id,
        text=text,
        index=index,
        metadata={"source": f"{doc_id}.txt"},
    )


class StubEmbedder:
    """假 Embedder：返回定长常量向量，记录收到过哪些文本。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.texts: list[str] = []
        self.closed = False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.5] * self.dim for _ in texts]

    async def aclose(self) -> None:
        self.closed = True


class StubReranker:
    """假重排器：把候选倒过来（分数随下标递增），并记录候选数量。"""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []
        self.closed = False

    async def rerank(
        self, query: str, texts: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        self.seen.append(list(texts))
        pairs = [(index, float(index)) for index in range(len(texts))]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:top_n] if top_n is not None else pairs

    async def aclose(self) -> None:
        self.closed = True


class StubStore:
    """假向量库：实现 ``VectorStore`` 协议，记录所有调用，不连任何数据库。

    关键点：它**不依赖 pymilvus**，也不需要模拟一致性、schema、索引这些
    数据库内部的东西。因为上层只依赖协议，测试就只需要一个几十行的替身。
    """

    def __init__(self, hits: list[ScoredChunk] | None = None) -> None:
        self.calls: list[str] = []
        self.stored: list[Chunk] = []
        self.deleted: list[str] = []
        self.closed = False
        self._hits = hits or []

    async def ensure_collection(self) -> None:
        self.calls.append("ensure_collection")

    async def upsert_chunks(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        self.calls.append("upsert_chunks")
        if len(chunks) != len(vectors):
            raise ValueError("chunk 数和向量数不一致")
        self.stored.extend(chunks)
        return len(chunks)

    async def delete_document(self, doc_id: str) -> int:
        self.calls.append("delete_document")
        self.deleted.append(doc_id)
        return 0

    async def flush(self) -> None:
        self.calls.append("flush")

    async def count(self) -> int:
        return len(self.stored)

    async def drop(self) -> None:
        self.calls.append("drop")
        self.stored.clear()

    async def search(
        self, query_vector: Sequence[float], *, top_k: int = 5, filter_expr: str = ""
    ) -> list[ScoredChunk]:
        self.calls.append("search")
        return self._hits[:top_k]

    async def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int = 5,
        filter_expr: str = "",
        candidate_k: int | None = None,
    ) -> list[ScoredChunk]:
        self.calls.append("hybrid_search")
        return self._hits[:top_k]

    async def aclose(self) -> None:
        self.closed = True
