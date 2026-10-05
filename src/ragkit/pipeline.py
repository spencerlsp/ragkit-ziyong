"""M9：门面 API —— 把整条流水线打包成一个对象。

在此之前，跑一次完整流程要自己把五个组件串起来：

    document = await parse_file(path)
    cleaned = clean_document(document)
    chunks = split_document(cleaned)
    await ingest_chunks(chunks, embedder=..., indexer=...)
    hits = await retriever.retrieve("问题")

现在收成两步：

    async with Ragkit() as rag:
        await rag.ingest(["docs/员工手册.md", "docs/报销制度.md"])
        hits = await rag.query("出差住宿费上限是多少", mode="hybrid", rerank=True)

**注意这不是「把细节藏起来」** —— 每个组件仍然可以注入替换
（``embedder=`` / ``indexer=`` / ``reranker=`` / ``splitter=``），
测试、定制、换实现都靠它。门面只是把**常见用法**缩短到不需要看文档。

这也是为什么 ``Ragkit`` 里几乎没有业务逻辑：
它只负责「按顺序调用」和「管理生命周期」。
真正的东西仍然在各自那一层里 —— **门面薄，是因为下层厚。**
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .embedding import Embedder, create_embedder
from .indexing import MilvusIndexer, VectorStore, ingest_chunks
from .parsing import parse_files
from .processing import clean_document
from .retrieval import Retriever
from .retrieval.reranker import Reranker, create_reranker
from .schemas import Chunk, ScoredChunk
from .splitting import RecursiveSplitter, Splitter

__all__ = ["IngestResult", "Ragkit"]


class IngestResult(BaseModel):
    """一次入库的结果。

    为什么把三个数字都返回，而不是只返回「写入了多少条」：
        「3 个文件 → 42 个 chunk → 42 条记录」这三个数一旦对不上，
        就说明中间某一步吞了东西：
          * files 多于 chunks —— 有文件解析出来是空的（或全是扫描件）；
          * chunks 多于 written —— 写入阶段丢数据了。
        只给一个总数，你什么都看不出来。
    """

    files: int
    chunks: int
    written: int
    # 入库的文档 ID。为什么要返回它：**这是你之后删除这批数据的唯一凭据**。
    # 不给的话，用户想删掉刚导进去的那篇文档，只能自己去算 stable_id(path, text)。
    doc_ids: list[str] = Field(default_factory=list)


class Ragkit:
    """把解析 / 切分 / 向量化 / 入库 / 检索串成两个方法。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        indexer: VectorStore | None = None,
        reranker: Reranker | None = None,
        splitter: Splitter | None = None,
    ) -> None:
        self._s = settings or get_settings()

        # 「谁创建，谁销毁」—— 四个组件各记一个所有权标记。
        # 组装类里的这个模式你已经见过三次了（MilvusIndexer、Retriever、现在这里），
        # 每多一层组合就多一层要盯的地方。
        self._owns_embedder = embedder is None
        self._owns_indexer = indexer is None
        self._owns_reranker = reranker is None

        self._embedder = embedder or create_embedder(self._s)
        self._indexer = indexer or MilvusIndexer(self._s)
        self._reranker = reranker or create_reranker(self._s)

        # 切分器没有资源要关，所以不需要所有权标记。
        self._splitter = splitter or RecursiveSplitter(
            chunk_size=self._s.chunk_size, chunk_overlap=self._s.chunk_overlap
        )

        # Retriever 拿到的三个组件**都是注入的**，所以它一个都不会关 ——
        # 生命周期统一由 Ragkit 管。这也避免了「两层都以为对方会关」或者
        # 「两层都去关」的尴尬。
        self._retriever = Retriever(
            embedder=self._embedder,
            indexer=self._indexer,
            reranker=self._reranker,
            settings=self._s,
        )

    async def __aenter__(self) -> Ragkit:
        """支持 ``async with Ragkit() as rag:``。"""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """只关自己创建的组件。"""
        if self._owns_embedder:
            await self._embedder.aclose()
        if self._owns_indexer:
            await self._indexer.aclose()
        if self._owns_reranker:
            await self._reranker.aclose()

    async def ingest(
        self,
        paths: Sequence[str | Path],
        *,
        reset: bool = False,
        flush: bool = True,
    ) -> IngestResult:
        """把一批文件解析、切分、向量化、写进向量库。

        ``reset=True`` 会先删掉整张表 —— **改了 schema 之后必须用它**，
        因为 collection 建好之后字段是改不了的。

        重复调用是**幂等**的：入库时会先按 doc_id 删掉该文档的旧数据，
        所以同一份文件导入两次不会产生重复记录。
        """
        items = [Path(path) for path in paths]
        if not items:
            return IngestResult(files=0, chunks=0, written=0)

        if reset:
            await self._indexer.drop()

        # 解析是并发的（内部用 TaskGroup + Semaphore 限流）
        try:
            documents = await parse_files(items)
        except ExceptionGroup as exc:
            # ★ TaskGroup 会把子任务的异常**打包**成 ExceptionGroup（PEP 654）。
            # 这是个很容易踩的坑：调用方写 `except ParseError` 是抓不到的，
            # 必须写 `except* ParseError` —— 而那个语法很多人没见过，
            # 报错信息也难看懂（「unhandled errors in a TaskGroup」）。
            #
            # 门面的职责之一就是**把这种底层细节吸收掉**：
            #   * 只有一个错误时，把它单独抛出来，`except ParseError` 就够了；
            #   * 有多个错误时保留 ExceptionGroup —— 那时你确实需要看到全部，
            #     随便挑一个抛出去反而是掩盖信息。
            errors = exc.exceptions
            if len(errors) == 1:
                raise errors[0] from exc
            raise

        # 清洗和切分是纯 CPU 的同步操作，串行即可 ——
        # 它们不等待任何东西，套异步只会让代码更绕。
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._splitter.split(clean_document(document)))

        written = await ingest_chunks(
            chunks, embedder=self._embedder, indexer=self._indexer, flush=flush
        )
        return IngestResult(
            files=len(documents),
            chunks=len(chunks),
            written=written,
            doc_ids=[document.doc_id for document in documents],
        )

    async def delete_document(self, doc_id: str) -> int:
        """删掉某个文档的所有 chunk，返回删除条数。

        ``doc_id`` 从 :meth:`ingest` 的返回值里拿：

            result = await rag.ingest(["docs/手册.md"])
            await rag.delete_document(result.doc_ids[0])

        ⚠️ **doc_id 是「路径 + 内容」的哈希**（见 ``utils.stable_id``），
        所以文件内容一变，doc_id 就变了。这意味着：
        「改了文件之后想删掉旧数据」不能靠重新解析这个文件 ——
        新解析出来的 id 和入库时的那个对不上。
        正确做法是保留 ingest 返回的 doc_ids，或者直接整表重建（``drop()`` + 重新 ``ingest()``）。
        """
        return await self._indexer.delete_document(doc_id)

    async def query(
        self,
        text: str,
        *,
        top_k: int | None = None,
        mode: Literal["dense", "hybrid"] = "dense",
        rerank: bool = False,
        rerank_candidates: int | None = None,
        score_threshold: float | None = None,
        filter_expr: str = "",
        total_timeout: float | None = None,
    ) -> list[ScoredChunk]:
        """检索与 text 最相关的片段，按相关度降序返回。

        参数含义全部沿用 :meth:`Retriever.retrieve` —— 这里只是转发，
        故意不做任何加工。**门面里多加一层「贴心的默认值」，
        就是在制造另一套需要单独记的规则。**
        """
        return await self._retriever.retrieve(
            text,
            top_k=top_k,
            mode=mode,
            rerank=rerank,
            rerank_candidates=rerank_candidates,
            score_threshold=score_threshold,
            filter_expr=filter_expr,
            total_timeout=total_timeout,
        )

    async def count(self) -> int:
        """返回向量库里的记录数（近似值，用于健康检查）。"""
        return await self._indexer.count()

    async def drop(self) -> None:
        """删掉整张表。

        单独暴露它是有意的：**重新导入前先清空**是排查问题时最常用的动作 ——
        怀疑数据脏了、schema 变了、切分参数改了，第一反应都是「推倒重来」。
        """
        await self._indexer.drop()
