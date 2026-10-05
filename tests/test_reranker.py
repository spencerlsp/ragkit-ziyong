"""重排客户端的测试。

全部离线：用 httpx.MockTransport 假装 /rerank 接口。

⚠️ 但这里的**假响应不是凭空想象的** —— 真实返回结构是我拿硅基流动的接口
实测过之后照着写的。凭想象造 mock 是离线测试最常见的坑：
你会测出一堆「符合自己想象」的绿色，然后在真环境里全线崩。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from ragkit.config import Settings
from ragkit.errors import RetrievalError
from ragkit.retrieval import HttpReranker, create_reranker

Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "rerank_base_url": "https://fake.test/v1",
        "rerank_api_key": "sk-test",
        "rerank_model": "fake-reranker",
        "max_retries": 2,
        "retry_base_delay": 0.01,
        "request_timeout": 5.0,
        "max_concurrency": 4,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://fake.test/v1"
    )


def rerank_response(pairs: list[tuple[int, float]]) -> dict[str, object]:
    """按**真实**结构拼响应：results 里每项带 index 和 relevance_score。"""
    return {
        "id": "fake",
        "results": [
            {"index": index, "document": None, "relevance_score": score} for index, score in pairs
        ],
    }


async def test_rerank_returns_index_score_pairs() -> None:
    """返回 (原始下标, 分数)，不是文本 —— 原文我们自己去取。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=rerank_response([(2, 0.9), (0, 0.5), (1, 0.1)]))

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        ranked = await reranker.rerank("q", ["甲", "乙", "丙"])
    finally:
        await reranker.aclose()

    assert ranked == [(2, 0.9), (0, 0.5), (1, 0.1)]


async def test_rerank_sorts_defensively() -> None:
    """服务端返回乱序时，必须自己再排一遍。

    真实的硅基流动接口返回的**确实是**降序的（实测过）。
    但「假设服务端排好了」一旦不成立，后果是「重排之后结果反而更差」——
    这种 bug 极难发现。排序是 O(n log n) 的廉价操作，这个保险值得买。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=rerank_response([(0, 0.1), (1, 0.9)]))

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        ranked = await reranker.rerank("q", ["甲", "乙"])
    finally:
        await reranker.aclose()

    assert ranked == [(1, 0.9), (0, 0.1)]


async def test_rerank_sends_expected_payload() -> None:
    """请求体要对：模型名、query、documents，以及不要求回传文档。"""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=rerank_response([(0, 1.0)]))

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        await reranker.rerank("问题", ["文档一", "文档二"], top_n=1)
    finally:
        await reranker.aclose()

    payload = seen[0]
    assert payload["model"] == "fake-reranker"
    assert payload["query"] == "问题"
    assert payload["documents"] == ["文档一", "文档二"]
    # 我们手上已经有原文，不需要服务端回传（省带宽）
    assert payload["return_documents"] is False
    assert payload["top_n"] == 1


async def test_rerank_empty_documents_makes_no_request() -> None:
    """空候选直接返回空，一次请求都不发。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=rerank_response([]))

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        assert await reranker.rerank("q", []) == []
    finally:
        await reranker.aclose()

    assert calls == 0


async def test_rerank_rejects_out_of_range_index() -> None:
    """下标越界必须拦住 —— 一旦错位，你拿到的就是「别处的文本」。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=rerank_response([(99, 0.9)]))

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        with pytest.raises(RetrievalError):
            await reranker.rerank("q", ["甲", "乙"])
    finally:
        await reranker.aclose()


async def test_rerank_rejects_missing_results() -> None:
    """返回体里没有 results 时要翻译成 RetrievalError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x"})

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        with pytest.raises(RetrievalError):
            await reranker.rerank("q", ["甲"])
    finally:
        await reranker.aclose()


async def test_rerank_retries_on_429() -> None:
    """限流要重试 —— 这套逻辑复用自 JsonClient，重排这边也该生效。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=rerank_response([(0, 0.9)]))

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        ranked = await reranker.rerank("q", ["甲"])
    finally:
        await reranker.aclose()

    assert ranked == [(0, 0.9)]
    assert calls == 2


async def test_rerank_does_not_retry_on_401() -> None:
    """key 错就立刻失败，不做无谓的重试。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "bad key"})

    reranker = HttpReranker(make_settings(), client=make_client(handler))
    try:
        with pytest.raises(RetrievalError):
            await reranker.rerank("q", ["甲"])
    finally:
        await reranker.aclose()

    assert calls == 1


async def test_aclose_does_not_close_injected_client() -> None:
    """注入的客户端不该被关（谁创建谁销毁）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=rerank_response([(0, 1.0)]))

    client = make_client(handler)
    reranker = HttpReranker(make_settings(), client=client)

    await reranker.aclose()

    assert not client.is_closed

    await client.aclose()


async def test_create_reranker_carries_settings() -> None:
    """工厂函数要把配置带进来 —— 尤其是认证头。"""
    reranker = create_reranker(make_settings())
    try:
        assert isinstance(reranker, HttpReranker)
        assert reranker._http._headers["Authorization"] == "Bearer sk-test"
    finally:
        await reranker.aclose()
