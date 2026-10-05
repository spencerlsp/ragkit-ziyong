"""检索层的测试。

同样全部离线：Milvus 用假客户端注入，Embedding 用假 embedder。
检索逻辑（分数方向、阈值过滤、Top-K 解析、超时）本来就都在 Python 侧，
不需要真数据库就能完整验证。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from pymilvus import MilvusException, RRFRanker

from ragkit.config import Settings
from ragkit.errors import RetrievalError
from ragkit.indexing import MilvusIndexer
from ragkit.indexing.schema import SPARSE_FIELD, VECTOR_FIELD
from ragkit.retrieval import Retriever, doc_filter

# ---------------------------------------------------------------------------
# 夹具和辅助
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    """完全受控的 Settings：不读 .env。"""
    base: dict[str, object] = {
        "_env_file": None,
        "openai_embedding_dim": 4,
        "milvus_uri": "http://fake-milvus:19530",
        "milvus_collection": "ragkit_test",
        "top_k": 3,
        "request_timeout": 5.0,
        "max_concurrency": 4,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_hit(
    chunk_id: str,
    score: float,
    *,
    doc_id: str = "d1",
    text: str = "某段内容",
    index: int = 0,
) -> dict[str, Any]:
    """造一条 Milvus 的命中（形状和真服务端一致）。"""
    return {
        "chunk_id": chunk_id,
        "distance": score,
        "entity": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "chunk_index": index,
            "source": f"{doc_id}.txt",
        },
    }


class FakeMilvusClient:
    """假的 Milvus 客户端：支持 search，记录调用。"""

    def __init__(self, hits: list[list[dict[str, Any]]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.collections: set[str] = set()
        self.hits = hits or []
        self.closed = False
        self.fail_on: str | None = None

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        self.calls.append((name, payload))

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            raise MilvusException(message=f"假的 {name} 失败")

    def payload_for(self, name: str) -> dict[str, Any]:
        for call_name, payload in self.calls:
            if call_name == name:
                return payload
        raise AssertionError(f"没有调用过 {name}")

    def names(self) -> list[str]:
        """按调用顺序返回方法名列表 —— 用来断言「调没调」「谁先谁后」。"""
        return [name for name, _ in self.calls]

    async def has_collection(self, collection_name: str, **kwargs: Any) -> bool:
        self._record("has_collection", {"collection_name": collection_name})
        await asyncio.sleep(0)
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        self._record("create_collection", {"collection_name": collection_name, **kwargs})
        await asyncio.sleep(0)
        self.collections.add(collection_name)

    async def search(
        self,
        collection_name: str,
        data: Any = None,
        anns_field: str | None = None,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        self._record(
            "search",
            {
                "collection_name": collection_name,
                "data": data,
                "anns_field": anns_field,
                "filter": filter,
                "limit": limit,
                "output_fields": output_fields,
            },
        )
        await asyncio.sleep(0)
        self._maybe_fail("search")
        return self.hits

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
        return self.hits

    async def close(self) -> None:
        self._record("close", {})
        self.closed = True


class StubEmbedder:
    """假 Embedder：返回定长向量；可以故意变慢，用来测超时。"""

    def __init__(self, dim: int = 4, delay: float = 0.0) -> None:
        self.dim = dim
        self.closed = False
        self._delay = delay
        self.texts: list[str] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.texts.extend(texts)
        return [[0.5] * self.dim for _ in texts]

    async def aclose(self) -> None:
        self.closed = True


class StubReranker:
    """假重排器：把候选**倒过来**，并记录收到了多少条候选。

    用「倒过来」而不是写死顺序，是因为候选数量会随参数变化 ——
    倒序在任何长度下都是确定的，好断言。
    """

    def __init__(self) -> None:
        self.seen: list[list[str]] = []
        self.closed = False

    async def rerank(
        self, query: str, texts: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        self.seen.append(list(texts))
        # 分数随下标递增，所以排序后就是倒序
        pairs = [(index, float(index)) for index in range(len(texts))]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:top_n] if top_n is not None else pairs

    async def aclose(self) -> None:
        self.closed = True


def make_indexer(client: FakeMilvusClient, **overrides: object) -> MilvusIndexer:
    return MilvusIndexer(make_settings(**overrides), client=client)


# ---------------------------------------------------------------------------
# MilvusIndexer.search
# ---------------------------------------------------------------------------


async def test_search_converts_hits_to_scored_chunks() -> None:
    """命中要转成 ScoredChunk，分数和字段都不能丢。"""
    client = FakeMilvusClient([[make_hit("c1", 0.9, text="命中内容", index=2)]])
    indexer = make_indexer(client)

    hits = await indexer.search([0.0] * 4, top_k=1)

    assert len(hits) == 1
    assert hits[0].score == 0.9
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].chunk.text == "命中内容"
    assert hits[0].chunk.index == 2
    assert hits[0].chunk.metadata["source"] == "d1.txt"


async def test_search_passes_filter_and_limit() -> None:
    """filter 表达式和 top_k 必须原样传给 Milvus。"""
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    await indexer.search([0.0] * 4, top_k=7, filter_expr='doc_id == "d1"')

    payload = client.payload_for("search")
    assert payload["limit"] == 7
    assert payload["filter"] == 'doc_id == "d1"'
    assert payload["data"] == [[0.0] * 4]  # 包了一层：Milvus 支持批量查询


async def test_search_does_not_request_vector_field() -> None:
    """取回的字段里绝不能包含 embedding。

    每条命中带 1024 个浮点数，会让返回体积暴涨几十倍，
    而检索结果里我们根本用不到向量。
    """
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    await indexer.search([0.0] * 4)

    assert VECTOR_FIELD not in client.payload_for("search")["output_fields"]


async def test_search_specifies_anns_field() -> None:
    """稠密检索必须**显式**说明搜哪个向量字段。

    这条是端到端实跑抓出来的：加了 BM25 稀疏字段之后，
    collection 里有了两个向量字段（embedding 和 sparse），
    不指定 anns_field 的话服务端会直接报
    ``multiple anns_fields exist, please specify a anns_field in search_params``。

    ⚠️ 关键在于**假客户端不会替你校验这个**：
    之前 165 条离线测试全绿，真库上却直接失败。
    所以必须把「参数传了没有」显式断言出来 ——
    这是 mock 唯一守不住、只能靠「把契约写成断言」来补的那类东西。
    """
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    await indexer.search([0.0] * 4)

    assert client.payload_for("search")["anns_field"] == VECTOR_FIELD


async def test_search_rejects_dim_mismatch() -> None:
    """查询向量维度不对时必须当场报错，并说清两边的维度。"""
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)  # 配置里是 4 维

    with pytest.raises(RetrievalError) as exc_info:
        await indexer.search([0.0] * 3)

    assert "3" in str(exc_info.value)
    assert "4" in str(exc_info.value)


async def test_search_rejects_bad_top_k() -> None:
    """top_k 必须为正数。"""
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    with pytest.raises(RetrievalError):
        await indexer.search([0.0] * 4, top_k=0)


async def test_search_translates_errors() -> None:
    """Milvus 的异常要翻译成 RetrievalError。"""
    client = FakeMilvusClient([[]])
    client.fail_on = "search"
    indexer = make_indexer(client)

    with pytest.raises(RetrievalError):
        await indexer.search([0.0] * 4)


async def test_search_rejects_malformed_hit() -> None:
    """服务端返回的命中缺字段时，要抛 RetrievalError，不能漏出 pydantic 的异常。

    ⚠️ 这条测试我第一版写错了：当时断言「命中缺 entity 时应该优雅降级成空文本」。
    那是错的 —— Chunk 的校验器从 M1 起就拒绝空白文本，
    所以「空文本」根本不是一个合法状态。
    **测试断言的行为必须和系统已有的契约一致，否则你测的是自己的想象。**

    正确的行为是：当场报错，而且报的是**我们自己的**异常类型。
    调用方统一写 `except RetrievalError`，任何内部异常漏出去都是在破坏契约。
    """
    client = FakeMilvusClient([[{"chunk_id": "c1", "distance": 0.5}]])
    indexer = make_indexer(client)

    with pytest.raises(RetrievalError):
        await indexer.search([0.0] * 4)


# ---------------------------------------------------------------------------
# MilvusIndexer.hybrid_search：两路召回 + RRF 融合
# ---------------------------------------------------------------------------


async def test_hybrid_search_builds_two_branches() -> None:
    """混合检索必须发两路：稠密那路收向量，关键词那路收**原始文本**。"""
    client = FakeMilvusClient([[make_hit("c1", 0.03)]])
    indexer = make_indexer(client)

    await indexer.hybrid_search([0.0] * 4, "你好", top_k=2)

    payload = client.payload_for("hybrid_search")
    dense, sparse = payload["reqs"]
    assert dense.anns_field == VECTOR_FIELD
    assert dense.data == [[0.0] * 4]
    assert sparse.anns_field == SPARSE_FIELD
    # ★ 关键词那一路传的是**文本**，不是向量 ——
    #   服务端会用 BM25 函数现场把查询文本转成稀疏向量。
    assert sparse.data == ["你好"]
    assert payload["limit"] == 2


async def test_hybrid_search_uses_rrf_ranker() -> None:
    """融合策略必须是 RRF（它只看排名，天然绕开两路分数的量纲差异）。"""
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    await indexer.hybrid_search([0.0] * 4, "你好")

    assert isinstance(client.payload_for("hybrid_search")["ranker"], RRFRanker)


async def test_hybrid_search_candidate_pool_exceeds_top_k() -> None:
    """每一路的候选数要比最终 top_k 大 —— 否则融合根本没有信息可用。

    RRF 只能看见每一路**给出的排名**。两路都只给 5 条的话，
    「稠密排第 100、关键词排第 1」的文档永远进不了融合 ——
    而它恰恰是混合检索最该捞回来的那种结果。
    """
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    await indexer.hybrid_search([0.0] * 4, "你好", top_k=3)

    reqs = client.payload_for("hybrid_search")["reqs"]
    assert [req.limit for req in reqs] == [12, 12]  # 3 * 4


async def test_hybrid_search_shares_filter_between_branches() -> None:
    """过滤条件必须同时作用在两路上，否则等于没过滤。"""
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    await indexer.hybrid_search([0.0] * 4, "你好", filter_expr='doc_id == "d1"')

    reqs = client.payload_for("hybrid_search")["reqs"]
    assert [req.expr for req in reqs] == ['doc_id == "d1"', 'doc_id == "d1"']


async def test_hybrid_search_rejects_blank_text() -> None:
    """关键词那一路需要原始文本；空文本算不出 BM25。"""
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    with pytest.raises(RetrievalError):
        await indexer.hybrid_search([0.0] * 4, "   ")


async def test_hybrid_search_rejects_dim_mismatch() -> None:
    """稠密那一路的维度检查不能省。"""
    client = FakeMilvusClient([[]])
    indexer = make_indexer(client)

    with pytest.raises(RetrievalError):
        await indexer.hybrid_search([0.0] * 3, "你好")


async def test_hybrid_search_translates_errors() -> None:
    """Milvus 的异常要翻译成 RetrievalError。"""
    client = FakeMilvusClient([[]])
    client.fail_on = "hybrid_search"
    indexer = make_indexer(client)

    with pytest.raises(RetrievalError):
        await indexer.hybrid_search([0.0] * 4, "你好")


# ---------------------------------------------------------------------------
# doc_filter
# ---------------------------------------------------------------------------


def test_doc_filter_builds_in_expression() -> None:
    """多个 doc_id 应该拼成 in [...] 表达式。"""
    assert doc_filter(["a", "b"]) == 'doc_id in ["a", "b"]'


def test_doc_filter_single() -> None:
    """单个 doc_id 也要合法。"""
    assert doc_filter(["only"]) == 'doc_id in ["only"]'


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


async def test_retrieve_returns_scored_chunks() -> None:
    """一次完整检索：向量化 + 查库 + 转换。"""
    client = FakeMilvusClient([[make_hit("c1", 0.9), make_hit("c2", 0.7)]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    hits = await retriever.retrieve("你好")

    assert [hit.chunk.chunk_id for hit in hits] == ["c1", "c2"]
    assert [hit.score for hit in hits] == [0.9, 0.7]


async def test_retrieve_uses_configured_top_k() -> None:
    """没传 top_k 时用配置里的值。"""
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(top_k=3),
    )

    await retriever.retrieve("你好")

    assert client.payload_for("search")["limit"] == 3


async def test_retrieve_explicit_top_k_wins() -> None:
    """显式传的 top_k 覆盖配置。"""
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(top_k=3),
    )

    await retriever.retrieve("你好", top_k=1)

    assert client.payload_for("search")["limit"] == 1


async def test_retrieve_does_not_treat_zero_top_k_as_missing() -> None:
    """top_k=0 不能被静默换成默认值 —— 这是 M5 falsy 坑的翻版。

    写成 `top_k or self._s.top_k` 的话，0 会被换成默认值，
    用户明确传 0 反而得到了 3 条结果，而且毫无提示。
    正确行为是：None 才代表「没提供」，0 应该报错。
    """
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(top_k=3),
    )

    with pytest.raises(RetrievalError):
        await retriever.retrieve("你好", top_k=0)


async def test_retrieve_rejects_blank_query() -> None:
    """空查询和纯空白查询都该被拦住，而且不该发任何请求。"""
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    for bad in ("", "   ", "\n\t"):
        with pytest.raises(RetrievalError):
            await retriever.retrieve(bad)

    assert client.calls == []


async def test_retrieve_filters_by_score_threshold() -> None:
    """低于阈值的命中要被丢掉（COSINE 是越大越相似）。"""
    client = FakeMilvusClient([[make_hit("c1", 0.9), make_hit("c2", 0.4), make_hit("c3", 0.75)]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    hits = await retriever.retrieve("你好", score_threshold=0.7)

    assert [hit.chunk.chunk_id for hit in hits] == ["c1", "c3"]


async def test_retrieve_translates_timeout() -> None:
    """整次检索超时要翻译成 RetrievalError，而不是漏出裸的 TimeoutError。

    这里让 embedder 睡 0.2 秒、总超时给 0.05 秒，必然触发。
    注意超时点在「向量化」这一步 —— 这正说明 asyncio.timeout
    是跨步骤生效的，不是只管最后那个数据库调用。
    """
    client = FakeMilvusClient([[make_hit("c1", 0.9)]])
    retriever = Retriever(
        embedder=StubEmbedder(delay=0.2),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    with pytest.raises(RetrievalError):
        await retriever.retrieve("你好", total_timeout=0.05)


async def test_retrieve_keeps_injected_components_open() -> None:
    """注入进来的组件不该被 retriever 关掉。"""
    client = FakeMilvusClient([[]])
    embedder = StubEmbedder()
    retriever = Retriever(
        embedder=embedder,
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    await retriever.retrieve("你好")
    await retriever.aclose()

    assert not embedder.closed
    assert not client.closed


async def test_retriever_closes_owned_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自己创建的组件必须被**真正**关掉，而不是「造了个协程就丢掉」。

    这条专门抓「写了 ``aclose()`` 但忘了 ``await``」：
    不 await 的话协程根本不会执行，客户端依然是开着的 ——
    程序不报错，只是安静地泄漏连接，直到某天句柄耗尽。
    这是 M2 里提醒过的 asyncio 第一大坑，在组合类里最容易复发。

    ⚠️ 为什么之前没抓到：另一条测试用的是**注入**的组件，
    ``_owns_*`` 全是 False，``aclose`` 里的两个分支根本没被走到。
    **「没被走到的分支等于没写」** —— 每个 if 分支都该有测试碰到它，
    尤其是「资源清理」这种平时看不见效果的分支。

    这里用 monkeypatch 把 ``create_embedder`` 换成替身，而不是去读
    ``retriever._embedder._http._client.is_closed``。两个理由：

      1) 读私有属性要穿透两层，**重构一次就断一次** ——
         后来把 HTTP 层抽成 JsonClient 时，这条测试就是这么断的；
      2) 我们真正想验证的是「aclose 被 await 了」，
         而不是「某个具体的 httpx 客户端被关了」。
         替身能记录前者，而且和实施细节解耦。
    """
    stub = StubEmbedder()
    monkeypatch.setattr("ragkit.retrieval.retriever.create_embedder", lambda settings=None: stub)
    indexer = MilvusIndexer(make_settings(), client=FakeMilvusClient())
    retriever = Retriever(indexer=indexer, settings=make_settings())

    await retriever.aclose()

    assert stub.closed


async def test_retriever_passes_settings_to_created_components() -> None:
    """注入 settings 时，自己创建的组件必须用同一份配置。

    这里不连网 —— 只是构造。但构造出来的 indexer 必须认得 8 维，
    说明 settings 确实传下去了。
    """
    retriever = Retriever(settings=make_settings(openai_embedding_dim=8))
    try:
        # 这里读私有属性是无奈的：Retriever 没有暴露 embedder 的公开 getter。
        # 但「配置有没有传下去」是真实存在的行为，而且只能这样观察 ——
        # 与其不测，不如带着注释测（并接受它对重构有一点耦合）。
        assert retriever._embedder.dim == 8
    finally:
        await retriever.aclose()


async def test_retrieve_hybrid_mode_uses_hybrid_search() -> None:
    """mode="hybrid" 时必须走 hybrid_search，而不是稠密的 search。"""
    client = FakeMilvusClient([[make_hit("c1", 0.03)]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    hits = await retriever.retrieve("你好", mode="hybrid")

    assert "hybrid_search" in client.names()
    assert "search" not in client.names()
    assert hits[0].chunk.chunk_id == "c1"


async def test_retrieve_defaults_to_dense_mode() -> None:
    """不传 mode 时保持原行为（稠密检索），向后兼容。"""
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    await retriever.retrieve("你好")

    assert "search" in client.names()
    assert "hybrid_search" not in client.names()


async def test_retrieve_hybrid_rejects_score_threshold() -> None:
    """混合模式 + 阈值 = 必须报错，绝不能静默给出错的结果。

    RRF 分数约 0~0.033，余弦相似度 0~1。拿 0.7 去过滤 RRF 分数，
    结果会被**全部**滤光；而用户只会看到「搜不到任何东西」，
    根本想不到是阈值量纲的问题。这种静默的错误结果比报错难查得多，
    所以宁可当场拒绝。
    """
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    with pytest.raises(RetrievalError):
        await retriever.retrieve("你好", mode="hybrid", score_threshold=0.7)

    assert client.calls == []  # 连请求都不该发


async def test_retrieve_rejects_unknown_mode() -> None:
    """mode 写错时要报错，而不是默默按稠密处理。"""
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    with pytest.raises(RetrievalError):
        await retriever.retrieve("你好", mode="bm25")  # type: ignore[arg-type]

    assert client.calls == []


async def test_retrieve_rerank_requires_a_reranker() -> None:
    """开了 rerank 却没给重排器时要报错，而不是默默跳过重排。

    「默默跳过」是最坏的选择：用户以为自己开了重排，
    指标却一点没动，然后花半天怀疑模型不行。
    """
    client = FakeMilvusClient([[]])
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        settings=make_settings(),
    )

    with pytest.raises(RetrievalError):
        await retriever.retrieve("你好", rerank=True)

    assert client.calls == []


async def test_retrieve_rerank_fetches_larger_candidate_pool() -> None:
    """重排前必须多召回一批候选，否则重排只能在 k 条里换顺序。

    k=3 时默认候选池是 max(3*4, 20) = 20。
    """
    hits = [[make_hit(f"c{i}", 0.9 - i * 0.01) for i in range(20)]]
    client = FakeMilvusClient(hits)
    stub = StubReranker()
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        reranker=stub,
        settings=make_settings(),
    )

    result = await retriever.retrieve("你好", top_k=3, rerank=True)

    # 召回阶段要了 20 条
    assert client.payload_for("search")["limit"] == 20
    # 重排器收到的也是 20 条候选
    assert len(stub.seen[0]) == 20
    # 最终只返回 3 条
    assert len(result) == 3


async def test_retrieve_rerank_reorders_candidates() -> None:
    """重排器倒序之后，返回顺序也要跟着倒过来。"""
    hits = [[make_hit("c0", 0.9), make_hit("c1", 0.8), make_hit("c2", 0.7)]]
    client = FakeMilvusClient(hits)
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        reranker=StubReranker(),
        settings=make_settings(),
    )

    result = await retriever.retrieve("你好", top_k=3, rerank=True, rerank_candidates=3)

    assert [hit.chunk.chunk_id for hit in result] == ["c2", "c1", "c0"]


async def test_retrieve_rerank_replaces_scores() -> None:
    """重排之后 score 必须换成重排分，不能还是余弦相似度。

    忘了换的话，后面的阈值过滤和排序展示全部基于错的量纲 ——
    而且不会有任何报错。
    """
    hits = [[make_hit("c0", 0.99), make_hit("c1", 0.98)]]
    client = FakeMilvusClient(hits)
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        reranker=StubReranker(),
        settings=make_settings(),
    )

    result = await retriever.retrieve("你好", top_k=2, rerank=True, rerank_candidates=2)

    # 假重排器给的分数是 [1.0, 0.0]，不是原来的 [0.99, 0.98]
    assert [hit.score for hit in result] == [1.0, 0.0]


async def test_retrieve_hybrid_with_rerank_allows_threshold() -> None:
    """混合模式 + 重排时，阈值重新变得有意义（分数被换成 0~1 的重排分）。"""
    hits = [[make_hit("c0", 0.9), make_hit("c1", 0.8)]]
    client = FakeMilvusClient(hits)
    retriever = Retriever(
        embedder=StubEmbedder(),
        indexer=make_indexer(client),
        reranker=StubReranker(),
        settings=make_settings(),
    )

    result = await retriever.retrieve(
        "你好",
        top_k=2,
        mode="hybrid",
        rerank=True,
        rerank_candidates=2,
        score_threshold=0.5,
    )

    # 重排分是 [1.0, 0.0]，阈值 0.5 只留下第一条
    assert len(result) == 1


def test_doc_filter_is_exported() -> None:
    """doc_filter 应该在 ragkit 顶层可用（M9 会用到）。"""
    import ragkit

    assert ragkit.doc_filter is doc_filter
