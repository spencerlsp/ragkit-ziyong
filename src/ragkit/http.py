"""带重试、限流、异常翻译的异步 JSON HTTP 客户端。

从 M4 的 ``OpenAICompatEmbedder`` 里抽出来的 —— 重排（rerank）要发
几乎一模一样的请求：POST 一个 JSON、拿回一个 JSON、429/5xx 要退避重试、
并发要限流。**这段逻辑里的每个边界条件都值得只修一次**，
所以抽出来共用，而不是抄第二遍。

它只认「JSON 进、JSON 出」的接口，不关心你是 embedding 还是 rerank ——
端点路径和请求体由调用方给。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .errors import RagkitError

__all__ = ["JsonClient"]


class JsonClient:
    """一个把「重试、限流、超时、异常翻译」都包好的异步 JSON 客户端。

    生命周期规则和别处一致：**谁创建，谁销毁**。
    外部注入的 ``client`` 不会被 ``aclose()`` 关掉。
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        max_retries: int,
        retry_base_delay: float,
        max_concurrency: int,
        error_cls: type[RagkitError],
        source: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )
        # 认证头挂在**请求**上，不是客户端上。
        # 客户端可能来自外部注入（测试、连接池复用），我们的密钥不该依赖它配合携带。
        self._headers = dict(headers)
        self._max_retries = max_retries
        self._retry_base = retry_base_delay
        self._error_cls = error_cls
        self._source = source
        self._sem = asyncio.Semaphore(max_concurrency)

    async def aclose(self) -> None:
        """只关自己创建的客户端。"""
        if self._owns_client:
            await self._client.aclose()

    async def _post_once(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发一次，不重试。

        闸门要覆盖**整个请求过程**，不是只包住「发起请求」那一下 ——
        否则请求还在飞、闸门已经放人了，限流就形同虚设。
        """
        async with self._sem:
            response = await self._client.post(path, json=payload, headers=self._headers)
            response.raise_for_status()
            return response.json()

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发一次带指数退避重试的 POST，返回解析后的 JSON。

        ★ 这段代码真正的一课是：**分清哪些错误该重试**。

          该重试（换个时间重试结果可能就变了）：
            * 超时 / 连接被重置 / DNS 抖动 → TimeoutException、TransportError
            * HTTP 429 → 被限流，等一下再来
            * HTTP 5xx → 服务端自己出问题

          不该重试（再试一万次结果也一样）：
            * 400 请求体格式错、401/403 key 不对、404 路径或模型名错、422 参数非法

          判断口诀：**换个时间重试，结果会不会变？**
          把 key 写错然后重试 3 次，你只是把「立刻失败」变成「7 秒后失败」，
          还给日志灌了一堆噪音。

        两个实现细节：
          * ``range(self._max_retries + 1)`` 里那个 +1 表示「首次尝试 + N 次重试」。
            写成 ``range(max_retries)`` 会少试一次。
          * ``min(base * 2**attempt, 30.0)`` 的上限不能省，
            不然重试次数一多，delay 会涨到几小时。
            指数退避是为了错开重试时刻（避免所有客户端同一秒一起冲上去）。

        生产环境还该加**抖动**：``delay * random.uniform(0.5, 1.0)``。
        这里先不加，但你得知道有这么个东西。
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._post_once(path, payload)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 and exc.response.status_code < 500:
                    raise self._error_cls(
                        f"请求被拒绝（HTTP {exc.response.status_code}）",
                        source=self._source,
                    ) from exc
                last_error = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt < self._max_retries:
                await asyncio.sleep(min(self._retry_base * (2**attempt), 30.0))

        raise self._error_cls(
            f"请求失败（已重试 {self._max_retries} 次）: {last_error}",
            source=self._source,
        ) from last_error
