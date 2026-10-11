"""检索策略注册表的测试。

全部离线：用 ``tests/fakes.py`` 里的替身把 ``Retriever`` 装起来，
既不连 Milvus 也不调 API。这正是「策略 = 名字 + 参数 + 一层包装」这个
设计的好处 —— 它自己不带资源，所以能直接测。
"""

from __future__ import annotations

import pytest
from fakes import StubEmbedder, StubReranker, StubStore, make_chunk

from ragkit.errors import RetrievalError
from ragkit.retrieval import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    Retriever,
    describe_strategies,
    resolve_strategy,
)
from ragkit.schemas import ScoredChunk


def hit(text: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(text), score=score)


def build(
    hits: list[ScoredChunk], *, reranker: StubReranker | None = None
) -> tuple[Retriever, StubStore]:
    """造一个组件全注入的 Retriever，并把 store 交出来方便断言。"""
    store = StubStore(hits)
    return Retriever(embedder=StubEmbedder(), indexer=store, reranker=reranker), store


# --------------------------------------------------------------------- 注册表


def test_expected_strategies_are_registered() -> None:
    """这几个名字是**对外契约**（会被写进 .env 和 MCP 客户端配置）。

    用子集判断而不是相等：加策略正是这份设计鼓励的事，
    但已有的名字不能在重构里悄悄消失 —— 那会让别人的配置当场失效。
    """
    assert {
        "dense",
        "hybrid",
        "hybrid_rerank",
        "section_aware",
        "section_aware_rerank",
    } <= set(STRATEGIES)


def test_registry_keys_match_strategy_names() -> None:
    """字典的键必须就是策略自己的名字，否则 ``resolve`` 会张冠李戴。"""
    for key, strategy in STRATEGIES.items():
        assert key == strategy.name


def test_default_strategy_exists() -> None:
    """默认值是字符串常量，写错了要到运行时才发现 —— 这里钉住它。"""
    assert DEFAULT_STRATEGY in STRATEGIES


def test_unknown_strategy_lists_the_alternatives() -> None:
    """名字写错时要把合法清单一起给出来 —— 模型拿到就能自己改对。"""
    with pytest.raises(RetrievalError) as exc_info:
        resolve_strategy("hybird")

    message = str(exc_info.value)
    assert "hybird" in message
    for name in STRATEGIES:
        assert name in message


def test_resolve_tolerates_surrounding_whitespace() -> None:
    """从配置文件读来的值常带空格，别为这个报错。"""
    assert resolve_strategy("  hybrid  ").name == "hybrid"


def test_resolve_rejects_empty_string() -> None:
    with pytest.raises(RetrievalError):
        resolve_strategy("")


def test_describe_marks_the_requested_default() -> None:
    """清单里的 ``*`` 要跟着**部署**的默认走，不能写死成代码里的那个。

    标错了，模型就会照着错的默认去挑策略。
    """
    text = describe_strategies(default="dense")
    assert "* dense：" in text
    assert f"* {DEFAULT_STRATEGY}：" not in text


def test_describe_lists_every_strategy() -> None:
    text = describe_strategies()
    for name in STRATEGIES:
        assert name in text
    assert f"* {DEFAULT_STRATEGY}：" in text


# --------------------------------------------------------------- 真的跑一遍


async def test_dense_strategy_only_does_vector_search() -> None:
    """``dense`` 不该碰 BM25 那一路。"""
    base, store = build([hit("内容 A")])

    out = await resolve_strategy("dense").search(base, "问题", top_k=3)

    assert store.calls == ["search"]
    assert len(out) == 1


async def test_hybrid_strategy_uses_hybrid_search() -> None:
    """``hybrid`` 要走两路召回那条接口。"""
    base, store = build([hit("内容 A")])

    await resolve_strategy("hybrid").search(base, "问题", top_k=3)

    assert store.calls == ["hybrid_search"]


async def test_rerank_strategy_calls_the_reranker() -> None:
    """带 rerank 的策略必须真的调重排器 —— 否则它和 hybrid 就没区别了。"""
    reranker = StubReranker()
    base, _ = build([hit("甲"), hit("乙"), hit("丙")], reranker=reranker)

    await resolve_strategy("hybrid_rerank").search(base, "问题", top_k=2)

    assert reranker.seen, "重排器一次都没被调用"
    assert reranker.seen[0] == ["甲", "乙", "丙"]


async def test_rerank_strategy_fails_loudly_without_a_reranker() -> None:
    """没配重排器时要报错。

    悄悄降级成普通混合检索是最坏的做法：结果看着正常，
    但你以为自己开着重排，于是照着「重排后的分数」下结论。
    """
    base, _ = build([hit("甲")], reranker=None)

    with pytest.raises(RetrievalError):
        await resolve_strategy("hybrid_rerank").search(base, "问题")


async def test_plain_strategy_does_not_touch_the_ranking() -> None:
    """``hybrid`` 不做后处理：有几条就返回几条。"""
    hits = [hit(f"[doc: X | section: Same]\n\n第 {i} 段正文。", 0.9) for i in range(4)]
    base, _ = build(hits)

    out = await resolve_strategy("hybrid").search(base, "问题", top_k=4)

    assert len(out) == 4


async def test_section_aware_strategy_applies_post_processing() -> None:
    """``section_aware`` 要真的套上后处理：同一节最多留 2 块（默认值）。

    同一批数据上 ``hybrid`` 返回 4 条、``section_aware`` 只返回 2 条 ——
    差出来的那两条就是这层后处理干的活。
    """
    hits = [hit(f"[doc: X | section: Same]\n\n第 {i} 段正文。", 0.9) for i in range(4)]
    base, _ = build(hits)

    out = await resolve_strategy("section_aware").search(base, "问题", top_k=4)

    assert len(out) == 2


async def test_section_aware_strategies_wrap_the_base() -> None:
    """两个 section 策略建立在同一个后处理层上，而且确实包了一层。"""
    base, _ = build([hit("[doc: X | section: S]\n\n正文。")])

    for name in ("section_aware", "section_aware_rerank"):
        assert resolve_strategy(name).wrap(base) is not base
