"""OpenAI 兼容的 Embedding 客户端 —— 手写 httpx，不用官方 SDK。

为什么手写：
  1) 端点协议很简单（就是一次 POST），手写能让你看清每一层；
  2) 你的目标之一是学异步，而 SDK 会把并发、重试、超时全藏起来；
  3) 顺手支持所有 OpenAI 兼容服务（硅基流动、DashScope、vLLM……），
     换供应商只改 base_url 和模型名。

HTTP 那一层（重试 / 限流 / 超时 / 异常翻译）已经抽到 ``ragkit.http.JsonClient``，
和重排客户端共用同一份实现 —— 见 retrieval/reranker.py。
真到生产想换回官方 SDK 也行：上层只认 Embedder 协议，换个实现文件而已。
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..config import Settings, get_settings
from ..errors import EmbeddingError
from ..http import JsonClient
from .base import Embedder

__all__ = ["OpenAICompatEmbedder", "create_embedder"]


class OpenAICompatEmbedder:
    """调用 ``POST {base_url}/embeddings`` 的客户端。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self.dim = self._s.openai_embedding_dim
        self._model = self._s.openai_embedding_model

        # 这是第三道维度防线（M1 的 Settings、这里、M5 建 schema 时）。
        # 换了 embedding 模型忘了同步改 OPENAI_EMBEDDING_DIM 的话，
        # 不在这里拦，就会一路走到 M5 写 Milvus 时才发现 —— 那时数据已经写了一半。
        if self.dim <= 0:
            raise EmbeddingError(f"向量维度必须为正数，收到 {self.dim}", source=self._model)

        self._http = JsonClient(
            base_url=self._s.openai_embedding_base_url,
            headers={
                # 注意 "Bearer " 后面有空格。少了它拼出来是 "Bearersk-xxx"，
                # 服务端回 401，而你会盯着 key 检查半天。
                "Authorization": ("Bearer " + self._s.openai_embedding_api_key.get_secret_value()),
            },
            timeout=self._s.request_timeout,
            max_retries=self._s.max_retries,
            retry_base_delay=self._s.retry_base_delay,
            max_concurrency=self._s.max_concurrency,
            error_cls=EmbeddingError,
            source=self._model,
            client=client,
        )

    async def aclose(self) -> None:
        """关掉底层 HTTP 客户端。注入进来的那份不会被关（谁创建谁销毁）。"""
        await self._http.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本转成向量。返回顺序严格对应输入顺序。"""
        if not texts:
            return []

        # 空白文本要**拦住**，不是过滤掉。
        # 调用方拿到的向量数量必须和输入文本数量严格一致 ——
        # 悄悄少给几条，调用方就会把 chunk 和向量错位配对，
        # 之后检索返回的永远是「别处的文本」，而且完全看不出异常。
        blanks = [i for i, text in enumerate(texts) if not text.strip()]
        if blanks:
            raise EmbeddingError(f"第 {blanks[:5]} 条文本是空白的，无法向量化", source=self._model)

        body = await self._http.post("/embeddings", {"model": self._model, "input": list(texts)})

        items = body.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            got = len(items) if isinstance(items, list) else "?"
            raise EmbeddingError(
                f"接口返回了 {got} 条向量，期望 {len(texts)} 条", source=self._model
            )

        # **别假设服务端返回顺序和输入一致。** OpenAI 协议里每条结果都带 index，
        # 但各家兼容实现的排序行为并不统一；按 index 排序是极廉价的保险。
        # 连 index 都不返回的服务会直接 KeyError 当场暴露 —— 那也比默默错位强。
        items = sorted(items, key=lambda item: item["index"])

        vectors: list[list[float]] = [list(item["embedding"]) for item in items]
        for i, vec in enumerate(vectors):
            if len(vec) != self.dim:
                raise EmbeddingError(
                    f"第 {i} 条向量是 {len(vec)} 维，但配置里写的是 {self.dim} 维；"
                    f"检查 OPENAI_EMBEDDING_DIM 是否和模型 {self._model} 匹配",
                    source=self._model,
                )
        return vectors


def create_embedder(settings: Settings | None = None) -> Embedder:
    """按配置造一个 Embedder。

    为什么不在这里做全局缓存（不像 get_settings 那样）：
        客户端持有 HTTP 连接池，是**有生命周期的资源**。
        全局缓存出去以后谁都不知道该何时关它，最后就是「永不关闭」。
        让调用方自己创建、自己 ``await aclose()``，责任清晰。
    """
    return OpenAICompatEmbedder(settings)
