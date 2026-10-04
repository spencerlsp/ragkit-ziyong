"""索引层的测试。

两层策略：

  * **单元测试**用 FakeMilvusClient 注入，不连任何数据库 —— 快、离线、随时能跑。
    能覆盖 schema 构建、双检锁、分批、删除顺序、异常翻译。
  * **集成测试**用环境变量开关，只有你真开着 Milvus 时才跑：
        $env:RAGKIT_TEST_MILVUS="1"; uv run pytest tests/test_indexing.py -q

schema 和 index_params 的构建完全在客户端完成，**根本不需要连库**，
所以那几条测试是最便宜也最稳的。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

import pytest
from pymilvus import DataType, MilvusException

from ragkit.config import Settings
from ragkit.errors import IndexingError
from ragkit.indexing import MilvusIndexer, build_index_params, build_schema, ingest_chunks
from ragkit.indexing.schema import (
    CHUNK_ID_FIELD,
    DOC_ID_FIELD,
    SPARSE_FIELD,
    VECTOR_FIELD,
)
from ragkit.schemas import Chunk
from ragkit.utils import stable_id

# ---------------------------------------------------------------------------
# 夹具和辅助
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    """完全受控的 Settings：不读 .env。"""
    base: dict[str, object] = {
        "_env_file": None,
        "openai_embedding_dim": 4,
        "milvus_uri": "http://fake-milvus:19530",
        "milvus_token": "",
        "milvus_collection": "ragkit_test",
        "milvus_batch_size": 2,
        "max_concurrency": 4,
        "request_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_chunk(text: str, doc_id: str = "d1", index: int = 0) -> Chunk:
    """造一个合法的 Chunk。"""
    return Chunk(
        chunk_id=stable_id(doc_id, str(index), text),
        doc_id=doc_id,
        text=text,
        index=index,
        metadata={"source": f"{doc_id}.txt"},
    )


class FakeMilvusClient:
    """假的 Milvus 客户端：记录所有调用，不连网。

    关键设计：has_collection / create_collection / upsert 里都插了一个
    ``await asyncio.sleep(0)``。

    **这不是装饰**。真实的 RPC 一定要等网络，等待期间事件循环会切到别的任务；
    如果假实现里没有任何 await，每个协程都会一口气跑完，
    那么「并发调用 ensure_collection」根本不会产生竞态，
    双检锁的测试就成了走过场 —— 你把锁删掉它照样绿。

    写并发相关的假实现时，一定要问自己：**我的假实现里有让出控制权的点吗？**
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.created: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.search_hits: list[list[dict[str, Any]]] = []
        self.collections: set[str] = set()
        self.closed = False
        self.fail_on: str | None = None

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        self.calls.append((name, payload))

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            raise MilvusException(message=f"假的 {name} 失败")

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    async def search(
        self,
        collection_name: str,
        data: Any = None,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        self._record(
            "search",
            {"collection_name": collection_name, "data": data, "filter": filter, "limit": limit},
        )
        await asyncio.sleep(0)
        self._maybe_fail("search")
        return self.search_hits

    async def hybrid_search(
        self,
        collection_name: str,
        reqs: list[Any] | None = None,
        ranker: Any = None,
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        self._record(
            "hybrid_search",
            {
                "collection_name": collection_name,
                "reqs": reqs,
                "ranker": ranker,
                "limit": limit,
                "output_fields": output_fields,
            },
        )
        await asyncio.sleep(0)
        self._maybe_fail("hybrid_search")
        return self.search_hits

    async def has_collection(self, collection_name: str, **kwargs: Any) -> bool:
        self._record("has_collection", {"collection_name": collection_name})
        await asyncio.sleep(0)  # 模拟一次 RPC 往返
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        self._record("create_collection", {"collection_name": collection_name, **kwargs})
        await asyncio.sleep(0)
        self._maybe_fail("create_collection")
        self.created.append({"collection_name": collection_name, **kwargs})
        self.collections.add(collection_name)

    async def upsert(self, collection_name: str, data: Any, **kwargs: Any) -> dict[str, int]:
        self._record("upsert", {"collection_name": collection_name, "data": data})
        await asyncio.sleep(0)
        self._maybe_fail("upsert")
        self.rows.extend(data)
        return {"upsert_count": len(data)}

    async def delete(self, collection_name: str, filter: str = "", **kwargs: Any) -> dict[str, int]:
        self._record("delete", {"collection_name": collection_name, "filter": filter})
        await asyncio.sleep(0)
        self._maybe_fail("delete")
        return {"delete_count": 0}

    async def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self._record("query", {"collection_name": collection_name, "filter": filter})
        await asyncio.sleep(0)
        return [{"count(*)": len(self.rows)}]

    async def drop_collection(self, collection_name: str, **kwargs: Any) -> None:
        self._record("drop_collection", {"collection_name": collection_name})
        await asyncio.sleep(0)
        self._maybe_fail("drop_collection")
        self.collections.discard(collection_name)

    async def close(self) -> None:
        self._record("close", {})
        self.closed = True


class StubEmbedder:
    """假 Embedder：返回确定性向量，不发请求。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []
        self.closed = False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(t))] * self.dim for t in texts]

    async def aclose(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# schema 与 index_params（完全不连库）
# ---------------------------------------------------------------------------


def test_schema_declares_all_fields() -> None:
    """默认 schema 声明 6 个字段（含 BM25 用的稀疏向量），名字都对得上。"""
    schema = build_schema(dim=8)
    assert len(schema.fields) == 6
    names = {field.name for field in schema.fields}
    assert names == {
        CHUNK_ID_FIELD,
        DOC_ID_FIELD,
        "text",
        "chunk_index",
        VECTOR_FIELD,
        SPARSE_FIELD,
    }


def test_schema_declares_bm25_function() -> None:
    """BM25 函数负责「text -> sparse」的自动转换，必须声明出来。"""
    schema = build_schema(dim=8)

    assert len(schema.functions) == 1
    fn = schema.functions[0]
    assert fn.name == "bm25_text_to_sparse"
    assert fn.type == 1  # FunctionType.BM25
    assert list(fn.input_field_names) == ["text"]
    assert list(fn.output_field_names) == [SPARSE_FIELD]


def test_schema_can_skip_sparse() -> None:
    """with_sparse=False 时回到纯稠密 schema（5 个字段、0 个函数）。

    保留这条路径是为了「从稠密起步、以后再升级混合」的场景，
    也是万一你的 Milvus 版本不支持 BM25 时的退路。
    """
    schema = build_schema(dim=8, with_sparse=False)
    assert len(schema.fields) == 5
    assert len(schema.functions) == 0


def test_schema_uses_configured_dim() -> None:
    """向量字段的维度必须等于传入的 dim。"""
    schema = build_schema(dim=1024)
    vector = next(field for field in schema.fields if field.name == VECTOR_FIELD)
    assert vector.params["dim"] == 1024


def test_schema_primary_key_is_varchar_chunk_id() -> None:
    """主键必须是字符串类型的 chunk_id —— 幂等写入全靠它。"""
    schema = build_schema(dim=8)
    primary = next(field for field in schema.fields if field.is_primary)
    assert primary.name == CHUNK_ID_FIELD
    assert primary.dtype == DataType.VARCHAR


def test_schema_rejects_bad_dim() -> None:
    """维度为 0 或负数时，应该在建 schema 之前就报错。"""
    for bad in (0, -1):
        with pytest.raises(IndexingError):
            build_schema(dim=bad)


def test_index_params_use_cosine() -> None:
    """度量方式必须是 COSINE（值越大越相似），它会一路传导到 M6 的排序逻辑。"""
    params = build_index_params()
    assert params[0].field_name == VECTOR_FIELD
    assert params[0].get_index_configs()["metric_type"] == "COSINE"


def test_index_params_include_bm25_sparse_index() -> None:
    """稀疏字段必须配 SPARSE_INVERTED_INDEX + BM25 —— Milvus 的固定搭配。"""
    params = build_index_params()

    assert len(params) == 2
    sparse = params[1]
    assert sparse.field_name == SPARSE_FIELD
    configs = sparse.get_index_configs()
    assert configs["index_type"] == "SPARSE_INVERTED_INDEX"
    assert configs["metric_type"] == "BM25"


def test_index_params_can_skip_sparse() -> None:
    """纯稠密模式下只建一个索引。"""
    assert len(build_index_params(with_sparse=False)) == 1


# ---------------------------------------------------------------------------
# ensure_collection：懒创建 + 双检锁
# ---------------------------------------------------------------------------


async def test_ensure_collection_creates_once() -> None:
    """第一次调用建表，第二次直接走缓存。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    await indexer.ensure_collection()
    await indexer.ensure_collection()

    assert len(client.created) == 1
    # 第二次不该再问服务端「有没有这张表」
    assert client.names().count("has_collection") == 1


async def test_ensure_collection_passes_schema_and_index() -> None:
    """建表时要把 schema 和 index_params 一起传进去（pymilvus 会自动建索引并 load）。"""
    client = FakeMilvusClient()
    await MilvusIndexer(make_settings(openai_embedding_dim=16), client=client).ensure_collection()

    created = client.created[0]
    assert created["collection_name"] == "ragkit_test"
    assert created["schema"] is not None
    assert created["index_params"] is not None
    vector = next(f for f in created["schema"].fields if f.name == VECTOR_FIELD)
    assert vector.params["dim"] == 16


async def test_ensure_collection_is_race_free() -> None:
    """10 个任务同时 ensure_collection，只能建一次表 —— 双检锁的核心价值。

    这条测试能红，靠的是 FakeMilvusClient 里的 ``await asyncio.sleep(0)``。
    没有那个让步点，10 个协程会依次一口气跑完，永远造不出竞态，
    你把锁删掉测试照样绿。
    """
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    await asyncio.gather(*(indexer.ensure_collection() for _ in range(10)))

    assert len(client.created) == 1


async def test_ensure_collection_translates_errors() -> None:
    """Milvus 抛的异常必须翻译成 IndexingError。"""
    client = FakeMilvusClient()
    client.fail_on = "create_collection"
    indexer = MilvusIndexer(make_settings(), client=client)

    with pytest.raises(IndexingError):
        await indexer.ensure_collection()


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


async def test_upsert_chunks_writes_expected_row() -> None:
    """写进去的行，字段名必须和 schema 完全对齐。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)
    chunk = make_chunk("你好世界", doc_id="doc-1", index=3)

    written = await indexer.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]])

    assert written == 1
    row = client.rows[0]
    assert row[CHUNK_ID_FIELD] == chunk.chunk_id
    assert row[DOC_ID_FIELD] == "doc-1"
    assert row["text"] == "你好世界"
    assert row["chunk_index"] == 3
    assert row[VECTOR_FIELD] == [0.1, 0.2, 0.3, 0.4]
    assert row["source"] == "doc-1.txt"  # 动态字段


async def test_upsert_chunks_batches_by_batch_size() -> None:
    """milvus_batch_size=2 时，5 条应该拆成 3 次 upsert 调用。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(milvus_batch_size=2), client=client)
    chunks = [make_chunk(f"第{i}段", index=i) for i in range(5)]
    vectors = [[float(i)] * 4 for i in range(5)]

    written = await indexer.upsert_chunks(chunks, vectors)

    assert written == 5
    sizes = sorted(len(payload["data"]) for name, payload in client.calls if name == "upsert")
    assert sizes == [1, 2, 2]
    assert len(client.rows) == 5


async def test_upsert_chunks_rejects_length_mismatch() -> None:
    """chunk 数和向量数不一致时必须当场报错，绝不能写进去。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    with pytest.raises(IndexingError):
        await indexer.upsert_chunks([make_chunk("a"), make_chunk("b")], [[0.0] * 4])

    assert client.rows == []


async def test_upsert_chunks_empty_returns_zero() -> None:
    """空输入直接返回 0，一次 RPC 都不发。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    assert await indexer.upsert_chunks([], []) == 0
    assert client.calls == []


async def test_upsert_chunks_translates_errors() -> None:
    """写入失败要翻译成 IndexingError。"""
    client = FakeMilvusClient()
    client.fail_on = "upsert"
    indexer = MilvusIndexer(make_settings(), client=client)

    with pytest.raises(IndexingError):
        await indexer.upsert_chunks([make_chunk("a")], [[0.0] * 4])


# ---------------------------------------------------------------------------
# delete / count / drop / aclose
# ---------------------------------------------------------------------------


async def test_delete_document_uses_doc_id_filter() -> None:
    """删除必须按 doc_id 过滤，而不是删全部。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    await indexer.delete_document("abc123")

    name, payload = client.calls[0]
    assert name == "delete"
    assert payload["filter"] == f'{DOC_ID_FIELD} == "abc123"'


async def test_count_queries_count_star() -> None:
    """计数走 query + count(*)，并且会先确保 collection 存在。"""
    client = FakeMilvusClient()
    client.rows = [{"a": 1}, {"a": 2}]
    indexer = MilvusIndexer(make_settings(), client=client)

    assert await indexer.count() == 2
    assert "create_collection" in client.names()


async def test_drop_resets_ready_cache() -> None:
    """删表之后必须清掉 _ready 缓存，否则下次会以为表还在。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    await indexer.ensure_collection()
    assert len(client.created) == 1

    await indexer.drop()
    await indexer.ensure_collection()

    assert len(client.created) == 2  # 重新建了一次


async def test_drop_is_idempotent() -> None:
    """表不存在时删表不该报错。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    await indexer.drop()

    assert "drop_collection" not in client.names()


async def test_drop_translates_errors() -> None:
    """删表失败也要翻译成 IndexingError，和这个类的其他方法保持一致。

    为什么这条重要：用户在门面 API 外面统一写 `except IndexingError`，
    如果某个方法漏了翻译，异常类型就漏出去了 ——
    而漏出去的是 pymilvus 的类型，用户根本不知道要去 import 它。
    「一致性」不是洁癖，它是接口契约的一部分。
    """
    client = FakeMilvusClient()
    client.collections.add("ragkit_test")  # 让 has_collection 返回 True
    client.fail_on = "drop_collection"
    indexer = MilvusIndexer(make_settings(), client=client)

    with pytest.raises(IndexingError):
        await indexer.drop()


async def test_aclose_only_closes_owned_client() -> None:
    """注入进来的客户端不该被关掉。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    await indexer.aclose()

    assert not client.closed


async def test_async_context_manager_closes_owned_client() -> None:
    """async with 退出时不该关掉注入进来的客户端。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    async with indexer:
        assert not client.closed

    assert not client.closed


# ---------------------------------------------------------------------------
# ingest_chunks：删旧 -> 向量化 -> 写入
# ---------------------------------------------------------------------------


async def test_ingest_deletes_before_upsert() -> None:
    """顺序必须是 delete 在 upsert 之前 —— 反了会把刚写的数据删掉。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)
    embedder = StubEmbedder()
    chunks = [
        make_chunk("第一段", doc_id="d1", index=0),
        make_chunk("第二段", doc_id="d1", index=1),
    ]

    written = await ingest_chunks(chunks, embedder=embedder, indexer=indexer)

    assert written == 2
    names = client.names()
    assert names.index("delete") < names.index("upsert")
    assert len(client.rows) == 2


async def test_ingest_dedupes_doc_id_deletes() -> None:
    """同一个 doc_id 的多个 chunk 只删一次。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)
    chunks = [make_chunk(f"第{i}段", doc_id="d1", index=i) for i in range(4)]

    await ingest_chunks(chunks, embedder=StubEmbedder(), indexer=indexer)

    assert client.names().count("delete") == 1


async def test_ingest_empty_returns_zero() -> None:
    """空输入直接返回 0，不该创建任何客户端。"""
    assert await ingest_chunks([]) == 0


async def test_ingest_keeps_injected_components_open() -> None:
    """注入进来的 embedder 和 indexer 都不该被关闭。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)
    embedder = StubEmbedder()

    await ingest_chunks([make_chunk("a")], embedder=embedder, indexer=indexer)

    assert not embedder.closed
    assert not client.closed


async def test_ingest_can_skip_delete() -> None:
    """replace_documents=False 时不该发删除请求（用于增量追加）。"""
    client = FakeMilvusClient()
    indexer = MilvusIndexer(make_settings(), client=client)

    await ingest_chunks(
        [make_chunk("a")],
        embedder=StubEmbedder(),
        indexer=indexer,
        replace_documents=False,
    )

    assert "delete" not in client.names()


# ---------------------------------------------------------------------------
# 集成测试：需要真 Milvus，默认跳过
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("RAGKIT_TEST_MILVUS") != "1",
    reason="需要真实 Milvus：先设 $env:RAGKIT_TEST_MILVUS='1'，并确保 .env 里的 MILVUS_URI 可连",
)
async def test_real_milvus_roundtrip() -> None:
    """真连一次 Milvus：建表 -> 写入 -> 计数 -> 删表。

    用独立的 collection 名（ragkit_it_test），不会碰你平时用的那个。
    这里读真实的 .env 拿 uri / token，只覆盖 collection 名。
    """
    settings = Settings(milvus_collection="ragkit_it_test")
    chunks = [make_chunk("第一段", doc_id="it-doc", index=0)]
    vectors = [[0.1] * settings.openai_embedding_dim]

    async with MilvusIndexer(settings) as indexer:
        await indexer.drop()
        await indexer.ensure_collection()
        assert await indexer.upsert_chunks(chunks, vectors) == 1
        assert await indexer.count() >= 0
        await indexer.drop()
