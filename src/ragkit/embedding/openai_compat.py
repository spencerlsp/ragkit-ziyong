"""OpenAI 兼容的 Embedding 客户端 —— 手写 httpx，不用官方 SDK。

为什么手写：
  1) 端点协议很简单（就是一次 POST），手写能让你看清每一层发生了什么；
  2) 你的目标之一是学异步，而 SDK 会把并发、重试、超时全藏起来；
  3) 顺手支持所有 OpenAI 兼容服务（硅基流动、DashScope、vLLM、Ollama……），
     换供应商只改 base_url 和模型名。

真到生产环境想换回 SDK 也行 —— 上层只认 Embedder 协议，换个实现文件而已。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from ..config import Settings, get_settings
from ..errors import EmbeddingError
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
        """TODO(你)：五件事。

        1) 解析配置：``self._s = settings or get_settings()``
           —— 允许注入配置，测试时就能塞一个受控的 Settings 进来。

        2) 把常用配置存成属性：

               self.dim = self._s.openai_embedding_dim
               self._model = self._s.openai_embedding_model
               self._max_retries = self._s.max_retries
               self._retry_base = self._s.retry_base_delay

           为什么只存这几个而不是全部：只存「方法体里真的会反复用到的」，
           其余留在 self._s 里按需取。存太多会让人分不清哪些是热配置。
           （``self.dim`` 是公开属性，协议要求的。）

        3) 校验维度：

               if self.dim <= 0:
                   raise EmbeddingError(f"向量维度必须为正数，收到 {self.dim}", source=self._model)

           这就是 M0 预告过的那个坑：换了 embedding 模型忘了同步改
           ``OPENAI_EMBEDDING_DIM``，会一路走到 M5 写 Milvus 时才报维度不匹配 ——
           那时候数据已经写了一半。**在这里拦住，成本几乎为零。**

        4) 建 HTTP 客户端（或者用注入进来的），再准备认证头：

               self._owns_client = client is None
               self._client = client or httpx.AsyncClient(
                   base_url=self._s.openai_embedding_base_url,
                   timeout=httpx.Timeout(self._s.request_timeout),
               )
               self._headers = {
                   "Authorization": "Bearer "
                   + self._s.openai_embedding_api_key.get_secret_value(),
               }

           要点：
             * ``self._owns_client`` 记住「这个客户端是不是我建的」。
               别人传进来的客户端不该由你关闭 —— 他可能还要继续用。
               **「谁创建，谁负责销毁」是资源管理的第一原则。**
             * **认证头挂在请求上，不挂在客户端上。** 上面是
               ``client or httpx.AsyncClient(...)``：如果把 headers 塞进 AsyncClient，
               只要有人注入了自己的 client，带 headers 的那个分支根本不会执行，
               认证就**静默消失**了。这个 bug 只有换掉 client 的那一刻才会暴露
               （测试、连接池复用……），生产里则表现为
               「我们做了个连接池优化，然后所有请求都 401」。
               分界原则：``base_url`` / 超时 / 连接池属于**客户端**（连接层面）；
               认证头 / 请求体属于**请求**（调用层面）。
             * ``get_secret_value()`` 是 SecretStr 唯一的取值方式。
               忘了调它，你会把 ``SecretStr('**********')`` 塞进请求头，
               服务端回你一个 401，而且报错信息里看不出任何异常（因为打印出来是掩码）。
             * ``httpx.Timeout`` 管的是**单次请求**的超时（连接阶段 + 读取阶段）。
               整批操作的截止时间是 embed_all 里用 wait_for 做的，层次不同。

        5) 建限流闸门：

               self._sem = asyncio.Semaphore(self._s.max_concurrency)

           **限流要放在客户端内部，不能放在调用方。**
           因为调用方完全可以绕过 embed_all，直接写个 for 循环调 embed() ——
           那样外部限流就彻底失效了。客户端自己握着闸门，
           不管谁怎么调，对服务端的压力都有上限。
        """
        self._s = settings or get_settings()

        self.dim = self._s.openai_embedding_dim
        self._model = self._s.openai_embedding_model
        self._max_retries = self._s.max_retries
        self._retry_base = self._s.retry_base_delay

        if self.dim <= 0:
            raise EmbeddingError(f"向量维度必须为正数，收到 {self.dim}", source=self._model)

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._s.openai_embedding_base_url,
            timeout=httpx.Timeout(self._s.request_timeout),
        )
        # 认证头挂在**请求**上，而不是客户端上：客户端可能来自外部注入，
        # 而我们的密钥不该依赖外部客户端配合携带。
        self._headers = {
            "Authorization": "Bearer " + self._s.openai_embedding_api_key.get_secret_value(),
        }

        self._sem = asyncio.Semaphore(self._s.max_concurrency)

    async def aclose(self) -> None:
        """TODO(你)：两行。

            if self._owns_client:
                await self._client.aclose()

        为什么必须有这个方法：httpx.AsyncClient 内部持有连接池和后台任务，
        不关会泄漏，程序退出时刷一屏 "Unclosed client session" 警告。

        顺带记住一个性能常识：**为每个请求新建客户端是新手最常见的性能杀手。**
        每建一次都要重做 TCP 握手 + TLS 握手（在国内访问境外 API 时这一步
        可能比请求本身还慢好几倍）。正确做法是全程复用一个客户端，
        这也是 M9 门面 API 里会用 ``async with`` 包住整条流水线的原因。
        """

        if self._owns_client:
            await self._client.aclose()

    async def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发一次请求，不重试。返回解析后的 JSON。

        TODO(你)：三步，全部写在 ``async with self._sem:`` 里面。

            async with self._sem:
                response = await self._client.post(
                    "/embeddings", json=payload, headers=self._headers
                )
                response.raise_for_status()
                return response.json()

        要点：
          * ``async with self._sem`` 必须覆盖**整个请求过程**，
            不是只包住创建请求那一下。如果只在外面包一层再退出，
            并发限制就形同虚设（请求还在飞，闸门已经放人了）。
          * ``raise_for_status()`` 把 4xx/5xx 变成 ``httpx.HTTPStatusError``。
            不调它的话，你拿到的是一段错误 JSON，然后在解析
            ``body["data"]`` 时才莫名其妙地 KeyError —— 错误地点离原因十万八千里。
          * 路径写 ``"/embeddings"`` 就够了，base_url 已经配在客户端里。
            httpx 会自动拼接；但如果你换了供应商后拿到 404，
            第一个要检查的就是 base_url 结尾有没有多余的斜杠。

        返回类型是 ``dict[str, Any]``：JSON 的结构我们无法在类型上表达，
        真正的校验在 embed() 里做。
        """
        async with self._sem:
            response = await self._client.post("/embeddings", json=payload, headers=self._headers)
            response.raise_for_status()
            return response.json()

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """带指数退避重试的请求。

        TODO(你)：按下面的骨架补全。

            last_error: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    return await self._post_once(payload)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 429 and exc.response.status_code < 500:
                        raise EmbeddingError(
                            # 4xx（429 除外）：请求本身有问题，重试一万次也一样
                            f"Embedding 请求被拒绝（HTTP {exc.response.status_code}）",
                            source=self._model,
                        ) from exc
                    last_error = exc
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc

                # 还有重试机会就退避等待
                if attempt < self._max_retries:
                    await asyncio.sleep(min(self._retry_base * (2**attempt), 30.0))

            raise EmbeddingError(
                f"调用 Embedding 接口失败（已重试 {self._max_retries} 次）: {last_error}",
                source=self._model,
            ) from last_error

        ★ 这段代码真正的一课是：**分清哪些错误该重试**。

          该重试（临时性故障，等一下可能就好了）：
            * 超时 / 连接被重置 / DNS 抖动 → httpx.TimeoutException、TransportError
            * HTTP 429 → 被限流，等一会儿再来
            * HTTP 5xx → 服务端自己出问题

          不该重试（再试一万次结果也一样，只是在浪费时间）：
            * HTTP 400 → 请求体格式错了
            * HTTP 401 / 403 → key 不对或没权限
            * HTTP 404 → 模型名写错了
            * HTTP 422 → 参数不合法

          判断口诀：**换个时间重试，结果会不会变？**
          会变 → 重试；不会变 → 立刻抛错，把问题暴露给用户。

          把 key 写错然后重试 3 次，你只是把「立刻失败」变成了「7 秒后失败」，
          还给日志灌了一堆没用的记录。

        两个实现细节：
          * ``for attempt in range(self._max_retries + 1)`` —— 加这个 1 很关键：
            它表示「首次尝试 + N 次重试」。写成 range(max_retries) 会少试一次。
          * 指数退避 ``base * 2**attempt``：1s、2s、4s、8s……
            服务端过载时，如果所有客户端都按固定间隔重试，会在同一秒一起冲上去
            （叫「惊群」）。指数退避拉开间隔，给服务端喘息时间。
            那个 ``min(..., 30.0)`` 上限也不能省，不然重试次数一多，
            delay 会涨到几小时。

        生产环境还应该加**抖动（jitter）**：``delay * random.uniform(0.5, 1.0)``，
        让不同客户端的重试时刻错开。这里先不加，但你得知道有这么个东西。
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._post_once(payload=payload)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 and exc.response.status_code < 500:
                    raise EmbeddingError(
                        # 4xx（429 除外）：请求本身有问题，重试一万次也一样
                        f"Embedding 请求被拒绝（HTTP {exc.response.status_code}）",
                        source=self._model,
                    ) from exc
                last_error = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            # 还有重试机会就退避等待
            if attempt < self._max_retries:
                await asyncio.sleep(min(self._retry_base * (2**attempt), 30.0))
        raise EmbeddingError(
            f"调用 Embedding 接口失败（已重试 {self._max_retries} 次）: {last_error}",
            source=self._model,
        ) from last_error

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本转成向量。

        TODO(你)：校验 → 请求 → 解析 → 再校验。

        1) 空输入直接返回 ``[]``（别浪费一次请求）。

        2) 挡住空白文本：

               blanks = [i for i, t in enumerate(texts) if not t.strip()]
               if blanks:
                   raise EmbeddingError(
                       f"第 {blanks[:5]} 条文本是空白的，无法向量化", source=self._model
                   )

           为什么不「悄悄过滤掉」再请求：
           调用方拿到的向量数量**必须**和传进来的文本数量一致（这是协议契约）。
           如果这里悄悄少返回几条，调用方会把 chunk 和向量错位配对 ——
           检索出来的结果是「别处的文本」，而这种静默的数据损坏
           可能几个月都发现不了。
           **宁可大声失败，不要悄悄丢数据。**

        3) 组装请求体并调用重试版请求：

               payload = {"model": self._model, "input": list(texts)}
               body = await self._post_with_retry(payload)

           ``list(texts)`` 那层转换是因为 json 序列化只认具体的 list，
           直接传 Sequence 有可能是个 tuple 或别的可迭代对象。

        4) 取出 data 并按 index 排序：

               items = body.get("data")
               if not isinstance(items, list) or len(items) != len(texts):
                   raise EmbeddingError(
                       f"接口返回了 {len(items) if isinstance(items, list) else '?'} 条向量，"
                       f"期望 {len(texts)} 条",
                       source=self._model,
                   )
               items = sorted(items, key=lambda item: item["index"])

           **别假设服务端返回顺序和输入一致。** OpenAI 的协议里每条结果都带
           ``index``，但各家兼容实现的排序行为并不统一。按 index 排序是极廉价的保险；
           如果某个服务连 index 都不返回，sorted 会直接 KeyError 当场暴露，
           也比默默错位强得多。

        5) 提取向量并校验维度：

               vectors = [list(item["embedding"]) for item in items]
               for i, vec in enumerate(vectors):
                   if len(vec) != self.dim:
                       raise EmbeddingError(
                           f"第 {i} 条向量是 {len(vec)} 维，但配置里写的是 {self.dim} 维；"
                           f"检查 OPENAI_EMBEDDING_DIM 是否和模型 {self._model} 匹配",
                           source=self._model,
                       )
               return vectors

           ``list(item["embedding"])`` 转成 list，是为了让返回值类型
           严格等于 ``list[list[float]]``（服务端给的可能是个数组，
           解析出来是 list 已经，但显式转换能挡住 httpx 或 json 库的类型漂移）。
        """
        if not texts:
            return []

        blanks = [i for i, t in enumerate(texts) if not t.strip()]
        if blanks:
            raise EmbeddingError(f"第 {blanks[:5]} 条文本是空白的，无法向量化", source=self._model)

        payload = {"model": self._model, "input": list(texts)}
        body = await self._post_with_retry(payload)

        items = body.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingError(
                f"接口返回了 {len(items) if isinstance(items, list) else '?'} 条向量，"
                f"期望 {len(texts)} 条",
                source=self._model,
            )
        items = sorted(items, key=lambda item: item["index"])

        vectors = [list(item["embedding"]) for item in items]
        for i, vec in enumerate(vectors):
            if len(vec) != self.dim:
                raise EmbeddingError(
                    f"第 {i} 条向量是 {len(vec)} 维，但配置里写的是 {self.dim} 维；"
                    f"检查 OPENAI_EMBEDDING_DIM 是否和模型 {self._model} 匹配",
                    source=self._model,
                )
        return vectors


def create_embedder(settings: Settings | None = None) -> Embedder:
    """按配置造一个 Embedder。已经写好了。

    为什么不在这里做全局缓存（不像 get_settings 那样）：
        客户端持有 HTTP 连接池，是**有生命周期的资源**。
        全局缓存出去以后，谁都不知道该在什么时候关它，
        最后就是「永不关闭」。让调用方自己创建、自己 ``await aclose()``，
        责任清晰。
    """
    return OpenAICompatEmbedder(settings)
