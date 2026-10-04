"""Embedding 层的测试。

好消息：**这些测试完全不需要网络，也不需要 API key。**
    * HashingEmbedder 本来就纯本地；
    * OpenAICompatEmbedder 用 httpx.MockTransport 注入一个假的 HTTP 层 ——
      请求会走进我们写的 handler，但根本不出网。

这也是「依赖注入」的回报：__init__ 允许传 client，测试就能整个换掉网络层。
如果当初在 __init__ 里写死了 httpx.AsyncClient(...)，这里就只能去 monkeypatch，
又脆又难读。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

import httpx
import pytest

from ragkit.config import Settings
from ragkit.embedding import Embedder, create_embedder, embed_all
from ragkit.embedding.hashing import HashingEmbedder
from ragkit.embedding.openai_compat import OpenAICompatEmbedder
from ragkit.errors import EmbeddingError

Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


# ---------------------------------------------------------------------------
# 夹具和辅助函数
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    """造一个完全受控的 Settings：不读 .env，所有值由这里决定。"""
    base: dict[str, object] = {
        "_env_file": None,  # 关掉 .env 读取
        "openai_embedding_base_url": "https://fake.test/v1",
        "openai_embedding_api_key": "sk-test",
        "openai_embedding_model": "fake-model",
        "openai_embedding_dim": 4,
        "openai_embedding_batch_size": 2,
        "max_concurrency": 2,
        "max_retries": 2,
        "retry_base_delay": 0.01,  # 测试里别真等 1 秒、2 秒
        "request_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_client(handler: Handler) -> httpx.AsyncClient:
    """造一个走 MockTransport 的客户端：请求不出网，直接进 handler。"""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://fake.test/v1",
    )


@asynccontextmanager
async def open_embedder(
    handler: Handler, **overrides: object
) -> AsyncIterator[OpenAICompatEmbedder]:
    """造一个走 MockTransport 的 embedder，退出时保证客户端被关掉。

    用 @asynccontextmanager 包一层，是为了让每个测试都短一点，
    并且**无论测试成功还是失败**都不会漏掉 aclose()。
    """
    client = make_client(handler)
    try:
        yield OpenAICompatEmbedder(make_settings(**overrides), client=client)
    finally:
        await client.aclose()


def embed_response(pairs: list[tuple[int, list[float]]]) -> dict[str, object]:
    """按 OpenAI 格式拼响应体。pairs 是 [(index, vector), ...]，顺序随你排。"""
    return {
        "data": [{"index": i, "embedding": v} for i, v in pairs],
        "model": "fake-model",
    }


def ok_handler(dim: int = 4) -> Handler:
    """一个永远成功的 handler，返回全 1 向量。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embed_response([(0, [1.0] * dim)]))

    return handler


class RecordingEmbedder:
    """假 Embedder：不发请求，只记录收到的批次，方便断言分片行为。"""

    def __init__(self, dim: int = 2, delay: float = 0.0) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []
        self._delay = delay

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.batches.append(list(texts))
        # 拿文本长度当向量值，方便断言「哪条向量对应哪条文本」
        return [[float(len(t))] * self.dim for t in texts]

    async def aclose(self) -> None:
        return None


def test_recording_embedder_satisfies_protocol() -> None:
    """RecordingEmbedder 应该满足 Embedder 协议（runtime_checkable 只查方法名）。"""
    assert isinstance(RecordingEmbedder(), Embedder)


# ---------------------------------------------------------------------------
# HashingEmbedder
# ---------------------------------------------------------------------------


async def test_hashing_is_deterministic() -> None:
    """同一个文本向量化两次，结果必须完全相同。

    这条防的是「用了 Python 内置的 hash()」：内置 hash 对 str 加了随机盐
    （PYTHONHASHSEED），同一个文本在不同进程里结果不一样 ——
    测试会时绿时红，而且换个进程就复现不出来。
    """
    embedder = HashingEmbedder(dim=8)
    first = await embedder.embed(["同一个文本"])
    second = await embedder.embed(["同一个文本"])
    assert first == second


async def test_hashing_differs_per_text() -> None:
    """不同文本的向量不能相同。"""
    embedder = HashingEmbedder(dim=8)
    cat, dog = await embedder.embed(["猫", "狗"])
    assert cat != dog


async def test_hashing_is_normalized() -> None:
    """每个向量的 L2 范数约等于 1（归一化之后长度信息就被消掉了）。"""
    embedder = HashingEmbedder(dim=8)
    vectors = await embedder.embed(["你好", "世界"])
    for vec in vectors:
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)


async def test_hashing_rejects_bad_dim() -> None:
    """dim 必须是正数。"""
    with pytest.raises(EmbeddingError):
        HashingEmbedder(dim=0)
    with pytest.raises(EmbeddingError):
        HashingEmbedder(dim=-1)


async def test_hashing_preserves_batch_order() -> None:
    """批量结果必须和逐条调用一一对应，不能错位。"""
    embedder = HashingEmbedder(dim=4)
    batch = await embedder.embed(["甲", "乙"])
    single_a = await embedder.embed(["甲"])
    single_b = await embedder.embed(["乙"])
    assert batch == [single_a[0], single_b[0]]


async def test_hashing_aclose_is_a_noop() -> None:
    """aclose 必须是一个可用的空实现。

    调用方会统一写 ``await embedder.aclose()``，它不该关心背后有没有真资源。
    如果这里留成 raise NotImplementedError，那么任何「统一收尾」的代码
    都会在运行时炸掉 —— 而且是在最不该出问题的地方（清理阶段）。
    """
    await HashingEmbedder(dim=4).aclose()


# ---------------------------------------------------------------------------
# embed_all：分片、并发、保序、超时
# ---------------------------------------------------------------------------


async def test_embed_all_empty() -> None:
    """空输入返回 []，而且一次请求都不该发。"""
    embedder = RecordingEmbedder()
    assert await embed_all(embedder, []) == []
    assert embedder.batches == []


async def test_embed_all_splits_into_batches() -> None:
    """5 条文本 + batch_size=2 → 3 批（2 + 2 + 1）。"""
    embedder = RecordingEmbedder()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = await embed_all(embedder, texts, batch_size=2)

    # 用 sorted 而不是直接比列表：gather 是并发的，批次**到达**顺序不保证。
    # 切片本身确定，但断言到达顺序就是在测实现细节了 ——
    # 并发测试里，凡是和顺序无关的断言都该写成顺序无关的形式。
    assert sorted(len(b) for b in embedder.batches) == [1, 2, 2]
    assert len(vectors) == 5


async def test_embed_all_preserves_order() -> None:
    """向量和文本必须按位置一一对应 —— M4 最重要的一条断言。

    向量和文本一旦错位，检索会返回「别的文本」，而表面上一切正常。
    这是最典型的静默数据损坏：不报错、不崩，只是答案悄悄变错。
    """
    embedder = RecordingEmbedder()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = await embed_all(embedder, texts, batch_size=2)

    assert [vec[0] for vec in vectors] == [float(len(t)) for t in texts]


async def test_embed_all_rejects_bad_batch_size() -> None:
    """batch_size 必须为正数。"""
    embedder = RecordingEmbedder()
    for bad in (0, -1):
        with pytest.raises(EmbeddingError):
            await embed_all(embedder, ["a"], batch_size=bad)


async def test_embed_all_total_timeout() -> None:
    """整批超时必须翻译成 EmbeddingError，而不是漏出裸的 TimeoutError。

    调用方只写 `except EmbeddingError` 就能兜住所有失败情况 ——
    这个「在边界处翻译异常」的套路 M2 已经练过了。
    """
    embedder = RecordingEmbedder(delay=0.2)
    with pytest.raises(EmbeddingError):
        await embed_all(embedder, ["a", "b"], batch_size=1, total_timeout=0.05)


# ---------------------------------------------------------------------------
# OpenAICompatEmbedder：用 MockTransport 测 HTTP 层
# ---------------------------------------------------------------------------


async def test_embed_parses_response() -> None:
    """正常响应应该被原样解析出来。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embed_response([(0, [1.0, 2.0, 3.0, 4.0])]))

    async with open_embedder(handler) as embedder:
        vectors = await embedder.embed(["你好"])

    assert vectors == [[1.0, 2.0, 3.0, 4.0]]


async def test_sends_bearer_token_header() -> None:
    """Authorization 头必须是 "Bearer " + key —— 注意 Bearer 后面那个空格。

    这条测试是 MockTransport 的「盲区补丁」：假的传输层不看协议细节，
    就算你把 header 写成 "Bearersk-xxx"，它照样返回 200。
    只有**主动把请求头读出来断言**，才能抓住这类错误 ——
    否则你会一路走到 M9 接真 API 才发现 401，然后对着代码看半天也看不出问题。

    结论：mock 能验证你的逻辑，验证不了「协议有没有写对」，
    后者必须显式断言。
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json=embed_response([(0, [1.0, 1.0, 1.0, 1.0])]))

    async with open_embedder(handler) as embedder:
        await embedder.embed(["你好"])

    assert seen == ["Bearer sk-test"]


async def test_embed_empty_input_makes_no_request() -> None:
    """空列表应该直接返回 []，一次请求都不发。

    注意 ``texts`` 的类型是 ``Sequence[str]``，它永远不可能是 None ——
    所以 ``if texts is None`` 是死代码。真正要处理的是**空列表**：
    它会让 payload 里的 input 变成 []，服务端通常直接回你一个 400。
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=embed_response([]))

    async with open_embedder(handler) as embedder:
        assert await embedder.embed([]) == []

    assert calls == 0


async def test_embed_sorts_by_index() -> None:
    """服务端返回乱序时，必须按 index 排好再返回。

    各家兼容实现的排序行为并不统一，所以「按 index 排序」是极廉价的保险。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # 故意乱序：index=1 排在前面
        return httpx.Response(
            200,
            json=embed_response([(1, [1.0, 1.0, 1.0, 1.0]), (0, [9.0, 9.0, 9.0, 9.0])]),
        )

    async with open_embedder(handler) as embedder:
        vectors = await embedder.embed(["第一", "第二"])

    assert vectors[0] == [9.0, 9.0, 9.0, 9.0]
    assert vectors[1] == [1.0, 1.0, 1.0, 1.0]


async def test_embed_rejects_count_mismatch() -> None:
    """返回条数和输入条数不符时必须报错，绝不能继续往下走。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embed_response([(0, [1.0, 1.0, 1.0, 1.0])]))

    async with open_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError):
            await embedder.embed(["第一", "第二"])


async def test_embed_rejects_dim_mismatch() -> None:
    """配置说 4 维、服务端返回 3 维 → 立刻报错。

    模拟的正是那个经典事故：换了 embedding 模型，
    忘了同步改 OPENAI_EMBEDDING_DIM。早在这里炸，比在 M5 写 Milvus 时炸好一百倍。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embed_response([(0, [1.0, 2.0, 3.0])]))

    async with open_embedder(handler) as embedder:  # make_settings 里 dim=4
        with pytest.raises(EmbeddingError):
            await embedder.embed(["你好"])


async def test_embed_rejects_blank_text() -> None:
    """空白文本要提前拦住，而且**一次请求都不该发**。

    为什么是拦住而不是过滤掉：返回的向量数量必须和输入文本数量严格一致。
    少给几条，调用方就会把 chunk 和向量错位配对 —— 静默的数据损坏。
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=embed_response([(0, [1.0, 1.0, 1.0, 1.0])]))

    async with open_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError):
            await embedder.embed(["ok", "   "])

    assert calls == 0


async def test_retries_on_429_then_succeeds() -> None:
    """429 是临时性的（被限流），等一会儿重试应该成功。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=embed_response([(0, [1.0, 1.0, 1.0, 1.0])]))

    async with open_embedder(handler) as embedder:
        vectors = await embedder.embed(["你好"])

    assert vectors == [[1.0, 1.0, 1.0, 1.0]]
    assert calls == 2


async def test_does_not_retry_on_401() -> None:
    """401 是永久性错误，重试没有任何意义。

    这条比「会重试」更重要：key 写错时重试 3 次，只是把「立刻失败」
    变成「7 秒后失败」，还把日志灌满噪音。
    判断标准：**换个时间重试，结果会不会变？** 401 不会变。
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    async with open_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError):
            await embedder.embed(["你好"])

    assert calls == 1


async def test_retries_exhausted_raises_embedding_error() -> None:
    """5xx 该重试，但重试次数用光之后要抛 EmbeddingError。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"error": "server exploded"})

    async with open_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError):
            await embedder.embed(["你好"])

    # make_settings 里 max_retries=2 → 1 次首试 + 2 次重试
    assert calls == 3


async def test_concurrency_is_bounded() -> None:
    """并发上限必须生效 —— M4 最有价值的一条异步测试。

    它能同时抓住两类常见 bug：
      * 忘了加 Semaphore → peak 会等于 5；
      * Semaphore 位置不对（比如只包住了「创建请求」那一下）→ peak 也会超过 2。

    只断言「结果对不对」的测试抓不到这两类问题，只有**测量并发度**才能。
    """
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)  # 假装网络很慢，制造重叠窗口
        active -= 1
        return httpx.Response(200, json=embed_response([(0, [1.0, 1.0, 1.0, 1.0])]))

    client = make_client(handler)
    try:
        embedder = OpenAICompatEmbedder(make_settings(max_concurrency=2), client=client)
        vectors = await embed_all(embedder, ["a", "b", "c", "d", "e"], batch_size=1)
    finally:
        await client.aclose()

    assert len(vectors) == 5
    assert peak <= 2
    # 这一条不能省：完全串行的实现也能让 peak <= 2 成立。
    # 同时断言下界，才是真的在验证「并发发生了，而且被限制住了」。
    assert peak == 2


async def test_aclose_does_not_close_injected_client() -> None:
    """注入进来的 client 不该被 embedder 关掉 ——「谁创建，谁负责销毁」。"""
    client = make_client(ok_handler())
    embedder = OpenAICompatEmbedder(make_settings(), client=client)

    await embedder.aclose()

    assert not client.is_closed

    await client.aclose()  # 我们自己造的，我们自己收尾


async def test_create_embedder_uses_settings() -> None:
    """create_embedder 应该把 Settings 里的维度带进来，并正确关闭自己的客户端。"""
    embedder = create_embedder(make_settings(openai_embedding_dim=7))
    try:
        assert embedder.dim == 7
    finally:
        await embedder.aclose()
