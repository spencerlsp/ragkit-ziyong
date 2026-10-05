"""门面 API（Ragkit）的测试。

全离线：向量库用 tests/fakes.py 里的 ``StubStore``，embedding 用 ``StubEmbedder``。
所以「解析 -> 清洗 -> 切分 -> 向量化 -> 入库」整条链路能在毫秒内跑完，
**不需要 Milvus，也不需要 API key**。

这能做到，靠的是把向量库抽成了 ``VectorStore`` 协议 ——
上层只依赖协议，测试就只需要一个几十行的替身，
而不是几百行的假 Milvus 客户端。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import StubEmbedder, StubReranker, StubStore, make_chunk

from ragkit import Ragkit
from ragkit.errors import ParseError
from ragkit.schemas import ScoredChunk


def write(path: Path, name: str, text: str) -> Path:
    target = path / name
    target.write_text(text, encoding="utf-8")
    return target


def build(**overrides: object) -> tuple[Ragkit, StubEmbedder, StubStore, StubReranker]:
    """造一个三个组件全注入的 Ragkit，方便测试观察。"""
    embedder, store, reranker = StubEmbedder(), StubStore(), StubReranker()
    rag = Ragkit(embedder=embedder, indexer=store, reranker=reranker)
    return rag, embedder, store, reranker


async def test_ingest_reports_files_chunks_written(tmp_path: Path) -> None:
    """入库要报出「几个文件 -> 几个 chunk -> 写入几条」这三个数。"""
    a = write(tmp_path, "a.md", "# 标题一\n\n这是第一份文档的内容，写长一点好切分。")
    b = write(tmp_path, "b.md", "# 标题二\n\n这是第二份文档的内容，也写长一点。")
    rag, _, store, _ = build()
    try:
        result = await rag.ingest([a, b])
    finally:
        await rag.aclose()

    assert result.files == 2
    assert result.chunks >= 2
    assert result.written == result.chunks
    assert len(store.stored) == result.chunks


async def test_ingest_empty_paths_is_a_noop() -> None:
    """空输入不该报错，也不该碰数据库。"""
    rag, _, store, _ = build()
    try:
        result = await rag.ingest([])
    finally:
        await rag.aclose()

    assert result.files == 0
    assert result.chunks == 0
    assert result.written == 0
    assert store.calls == []


async def test_ingest_reset_drops_before_writing(tmp_path: Path) -> None:
    """reset=True 必须先删表再写 —— 顺序反了就把刚写的数据删了。"""
    a = write(tmp_path, "a.md", "一些内容")
    rag, _, store, _ = build()
    try:
        await rag.ingest([a], reset=True)
    finally:
        await rag.aclose()

    assert store.calls[0] == "drop"
    assert "upsert_chunks" in store.calls
    assert store.calls.index("drop") < store.calls.index("upsert_chunks")


async def test_ingest_replaces_same_document(tmp_path: Path) -> None:
    """同一份文件导入两次，要先按 doc_id 删旧数据（幂等），而不是堆重复。"""
    a = write(tmp_path, "a.md", "同一份文件的内容")
    rag, _, store, _ = build()
    try:
        await rag.ingest([a])
        await rag.ingest([a])
    finally:
        await rag.aclose()

    # doc_id 被删了两次（每次导入删一次），对应同一个文档
    assert len(store.deleted) == 2
    assert store.deleted[0] == store.deleted[1]


async def test_ingest_flushes_by_default(tmp_path: Path) -> None:
    """入库结束要 flush，否则刚写的可能查不到（Milvus 是最终一致性的）。"""
    a = write(tmp_path, "a.md", "一些内容")
    rag, _, store, _ = build()
    try:
        await rag.ingest([a])
    finally:
        await rag.aclose()

    assert "flush" in store.calls


async def test_ingest_can_skip_flush(tmp_path: Path) -> None:
    """大批量导入时应该能关掉 flush，最后统一刷。"""
    a = write(tmp_path, "a.md", "一些内容")
    rag, _, store, _ = build()
    try:
        await rag.ingest([a], flush=False)
    finally:
        await rag.aclose()

    assert "flush" not in store.calls


async def test_ingest_propagates_parse_error(tmp_path: Path) -> None:
    """单个文件出错时，门面要把 ParseError **解包**出来。

    为什么要解包：parse_files 内部用 TaskGroup，而 TaskGroup 会把子任务的
    异常打包成 ExceptionGroup（PEP 654）。不解包的话，调用方写
    `except ParseError` 是**抓不到的** —— 必须写 `except* ParseError`，
    而那个语法很多人没见过，报错信息也难看懂。

    **门面存在的意义之一，就是让常见用法不需要知道这些底层细节。**
    """
    rag, _, _, _ = build()
    try:
        with pytest.raises(ParseError):
            await rag.ingest([tmp_path / "不存在.md"])
    finally:
        await rag.aclose()


async def test_ingest_keeps_exception_group_for_multiple_failures(tmp_path: Path) -> None:
    """多个文件同时坏掉时，保留 ExceptionGroup —— 那时你确实需要看到全部。

    只挑一个错误抛出去，会掩盖「另外两个文件也坏了」这个重要信息。
    什么时候解包、什么时候保留，取决于哪个行为对调用方更有用。
    """
    rag, _, _, _ = build()
    try:
        with pytest.raises(ExceptionGroup) as exc_info:
            await rag.ingest([tmp_path / "a.md", tmp_path / "b.md", tmp_path / "c.md"])
    finally:
        await rag.aclose()

    assert len(exc_info.value.exceptions) == 3


async def test_query_defaults_to_dense() -> None:
    """不传 mode 时走稠密检索。"""
    hits = [ScoredChunk(chunk=make_chunk("内容"), score=0.9)]
    embedder, store, reranker = StubEmbedder(), StubStore(hits), StubReranker()
    rag = Ragkit(embedder=embedder, indexer=store, reranker=reranker)
    try:
        result = await rag.query("问题")
    finally:
        await rag.aclose()

    assert store.calls == ["search"]
    assert len(result) == 1
    assert result[0].score == 0.9


async def test_query_forwards_mode_and_rerank() -> None:
    """mode 和 rerank 要一路传到最底层，不能被门面吃掉。"""
    hits = [ScoredChunk(chunk=make_chunk("内容"), score=0.5)]
    embedder, store, reranker = StubEmbedder(), StubStore(hits), StubReranker()
    rag = Ragkit(embedder=embedder, indexer=store, reranker=reranker)
    try:
        await rag.query("问题", mode="hybrid", rerank=True, rerank_candidates=5)
    finally:
        await rag.aclose()

    assert "hybrid_search" in store.calls
    assert "search" not in store.calls
    assert reranker.seen  # 重排器真的被调用了


async def test_query_forwards_filter(tmp_path: Path) -> None:
    """过滤条件要传下去（它决定「只看某个文档」这类功能）。"""
    hits: list[ScoredChunk] = []
    embedder, store, reranker = StubEmbedder(), StubStore(hits), StubReranker()
    rag = Ragkit(embedder=embedder, indexer=store, reranker=reranker)
    try:
        await rag.query("问题", filter_expr='doc_id == "d1"')
    finally:
        await rag.aclose()

    assert store.calls == ["search"]


async def test_count_and_drop() -> None:
    """count / drop 直接转发给向量库。"""
    rag, _, store, _ = build()
    try:
        store.stored.extend([make_chunk("a"), make_chunk("b")])
        assert await rag.count() == 2
        await rag.drop()
        assert await rag.count() == 0
    finally:
        await rag.aclose()

    assert "drop" in store.calls


async def test_aclose_does_not_close_injected_components() -> None:
    """注入进来的组件一个都不该被关。"""
    rag, embedder, store, reranker = build()

    await rag.aclose()

    assert not embedder.closed
    assert not store.closed
    assert not reranker.closed


async def test_aclose_closes_all_owned_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三个组件都由 Ragkit 自建时，三个都要被真正关掉。

    这条专门抓「漏写一个 await」—— 它在这类组合类里最容易复发
    （Retriever 当年就在这上面栽过一次）。
    """
    embedder, store, reranker = StubEmbedder(), StubStore(), StubReranker()
    monkeypatch.setattr("ragkit.pipeline.create_embedder", lambda settings=None: embedder)
    monkeypatch.setattr("ragkit.pipeline.MilvusIndexer", lambda settings=None: store)
    monkeypatch.setattr("ragkit.pipeline.create_reranker", lambda settings=None: reranker)

    rag = Ragkit()
    await rag.aclose()

    assert embedder.closed
    assert store.closed
    assert reranker.closed


async def test_context_manager_closes_owned_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``async with`` 退出时要自动收尾。"""
    store = StubStore()
    monkeypatch.setattr("ragkit.pipeline.create_embedder", lambda settings=None: StubEmbedder())
    monkeypatch.setattr("ragkit.pipeline.MilvusIndexer", lambda settings=None: store)
    monkeypatch.setattr("ragkit.pipeline.create_reranker", lambda settings=None: StubReranker())

    async with Ragkit():
        assert not store.closed

    assert store.closed
