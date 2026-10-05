"""检索层：query -> 向量 -> Milvus -> 一批 ScoredChunk。

这一层把两个组件组合起来（Embedder + MilvusIndexer），
所以「谁创建、谁负责销毁」的规矩在这里尤其重要：
组合场景比单组件更容易犯「顺手把别人的资源关掉」的错。

★ M6 的新异步知识点：``async with asyncio.timeout(...)``
    M4 用的是 ``asyncio.wait_for(coro, t)``，它只能包**一个** awaitable。
    而一次检索横跨两步（先向量化、再查库），
    用 ``asyncio.timeout`` 上下文管理器能包住一整段逻辑，更自然。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

from ..config import Settings, get_settings
from ..embedding import Embedder, create_embedder
from ..errors import RetrievalError
from ..indexing import MilvusIndexer
from ..indexing.schema import DOC_ID_FIELD
from ..schemas import ScoredChunk
from .reranker import Reranker

__all__ = ["Retriever", "doc_filter"]


def doc_filter(doc_ids: Sequence[str]) -> str:
    """构造「只在这些文档里检索」的 Milvus filter 表达式。

    TODO(你)：两行。

        quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
        return f"{DOC_ID_FIELD} in [{quoted}]"

    效果：``doc_filter(["a", "b"])`` -> ``doc_id in ["a", "b"]``

    ⚠️ 和 M5 的 delete 一样，拼接字符串是有注入风险的：
       如果某个 doc_id 里带了引号，整个表达式就被破坏了。
       我们的 doc_id 来自 stable_id，是十六进制串，天然安全；
       但只要来源变成用户输入，就必须先校验格式。

    为什么值得单独抽一个函数：
       filter 语法是 Milvus 特有的。散在业务代码里，
       等于让上层被数据库细节污染；收在一处，将来换向量库只改这里。
    """
    quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
    return f"{DOC_ID_FIELD} in [{quoted}]"


class Retriever:
    """组合 Embedder 和 MilvusIndexer，提供「文本进、片段出」的检索接口。"""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        indexer: MilvusIndexer | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        """TODO(你)：三件事。

        1) ``self._s = settings or get_settings()``

        2) 记录所有权，并创建缺失的组件：

               self._owns_embedder = embedder is None
               self._owns_indexer = indexer is None
               self._embedder = embedder or create_embedder(self._s)
               self._indexer = indexer or MilvusIndexer(self._s)

           **注意把 self._s 往下传。** 这样当你注入一份自定义 settings 时，
           两个子组件用的是**同一份配置** —— 否则会出现
           「embedder 按 1024 维算、indexer 也按 1024 维，但其实是两份不同的配置」
           这种只在特定路径下才暴露的诡异问题。
           构造参数一路贯穿下去，是组合类的基本修养。

        3) 不用在这里校验两个组件的维度是否一致 ——
           search() 里那道维度检查会替你抓住：如果 embedder 产出的维度和
           库里的不一致，报错信息会直接告诉你「查询向量 768 维，库里 1024 维」。
           **能在一个地方报清楚的错，就别在两处报。**
        """
        self._s = settings or get_settings()
        self._owns_embedder = embedder is None
        self._owns_indexer = indexer is None
        self._embedder = embedder or create_embedder(self._s)
        self._indexer = indexer or MilvusIndexer(self._s)
        # 重排器**不自动创建** —— 它需要单独的 API key 和模型配置，
        # 我们不想让「检索」默认就依赖一个可选能力。
        # 也正因为如此，它永远是外部注入的，aclose 时不关它。
        self._reranker = reranker

    async def __aenter__(self) -> Retriever:
        """支持 ``async with Retriever(...) as r:``。已经写好了。"""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """退出 async with 时自动收尾。已经写好了。"""
        await self.aclose()

    async def aclose(self) -> None:
        """TODO(你)：只关自己创建的那两个组件。

            if self._owns_embedder:
                await self._embedder.aclose()
            if self._owns_indexer:
                await self._indexer.aclose()

        顺序无所谓（两者互相独立），但两个都要判断所有权 ——
        漏掉任何一个，都会在「别人注入组件」的场景下把对方的东西关掉。
        """
        if self._owns_embedder:
            await self._embedder.aclose()
        if self._owns_indexer:
            await self._indexer.aclose()
        # 重排器永远是外部注入的，交给调用方收尾。

    @staticmethod
    async def _apply_rerank(
        reranker: Reranker,
        query: str,
        hits: Sequence[ScoredChunk],
        *,
        top_n: int,
    ) -> list[ScoredChunk]:
        """用重排器给候选重新打分，取前 top_n 条。

        两个要点：
          * 传给重排器的是**候选文本**，拿回来的是 **(原始下标, 分数)** ——
            我们用下标回指原来的 ScoredChunk，这样 chunk 的原对象
            （以及里面所有元数据）完好无损，只是分数被换掉了。
          * 重排后 score 的含义变了：从余弦相似度 / RRF 分
            变成**重排模型给的相关度分**。同一个 0.7 在两个阶段
            完全不是一个意思，别混着用。
        """
        ranked = await reranker.rerank(query, [hit.chunk.text for hit in hits], top_n=top_n)
        return [ScoredChunk(chunk=hits[index].chunk, score=score) for index, score in ranked]

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_expr: str = "",
        total_timeout: float | None = None,
        mode: Literal["dense", "hybrid"] = "dense",
        rerank: bool = False,
        rerank_candidates: int | None = None,
    ) -> list[ScoredChunk]:
        """检索与 query 最相关的片段，按相关度降序返回。

        ``mode`` 有两种：

          * ``"dense"``（默认）：只用向量检索。score 是余弦相似度，
            0~1，**越大越相关**。
          * ``"hybrid"``：向量 + BM25 关键词两路召回，RRF 融合。
            **score 是 RRF 融合分（量级约 0~0.033），和稠密模式不可比** ——
            所以这个模式下不允许传 ``score_threshold``（见下面第 3 步）。

        ``rerank=True`` 会在召回之后多加一步**精排**：
          * 召回阶段先多取一批候选（默认 ``max(top_k * 4, 20)`` 条）；
          * 用重排模型对这批候选重新打分，取前 ``top_k`` 条；
          * **返回的 score 变成重排模型的相关度分（0~1）**，
            既不是余弦相似度、也不是 RRF 分。

        为什么重排要多取候选：只把 top_k 条拿去重排，它只能在 k 条里换顺序，
        一点召回收益都拿不到。**重排的全部价值来自「把本该进前列、
        却被向量检索挤出去的那几条捞回来」** —— 所以候选池必须放大。

        TODO(你)：五步。

        1) 校验查询：

               query = query.strip()
               if not query:
                   raise RetrievalError("查询文本不能为空")

        2) 解析 top_k —— **注意别用 `or`**：

               k = top_k if top_k is not None else self._s.top_k

           这是 M5 那个 falsy 坑的翻版。写成 ``top_k or self._s.top_k`` 的话，
           传 0 会被静默换成默认值 —— 而 0 在这里是有意义的
           （虽然语义上非法，但应该报错，而不是偷偷改成 5）。
           **只有 None 才代表「调用方没提供」。**

        3) 用 ``asyncio.timeout`` 把「向量化 + 检索」当一个整体包起来：

               timeout = (
                   total_timeout if total_timeout is not None else self._s.request_timeout
               )
               try:
                   async with asyncio.timeout(timeout):
                       vectors = await self._embedder.embed([query])
                       if not vectors:
                           raise RetrievalError(
                               "Embedding 返回了空结果", source=query[:50]
                           )
                       hits = await self._indexer.search(
                           vectors[0], top_k=k, filter_expr=filter_expr
                       )
               except TimeoutError as exc:
                   raise RetrievalError(
                       f"检索超时（{timeout} 秒）", source=query[:50]
                   ) from exc

           ★ **这是 M6 的新知识点**，和 M4 的 wait_for 对比着记：

             * ``asyncio.wait_for(coro, t)`` 只能包**一个** awaitable。
               超时时它取消那个 coroutine。
             * ``async with asyncio.timeout(t):`` 能包**一整段逻辑** ——
               上面两步算一个整体，谁慢都算整次检索超时。
               而且超时触发时，正在等待的那一步会被**取消**
               （HTTP 请求会被真的断掉，而不是继续跑完白费）。

           为什么这里不用 wait_for：我们的截止时间是给「一次检索」的，
           它横跨两个 await。用 wait_for 就得把两步塞进一个嵌套协程，
           多一层缩进不说，读起来也更绕。

           （Python 3.11+ 才有 asyncio.timeout，我们正好是 3.11。）

        4) 按阈值过滤：

               if score_threshold is not None:
                   hits = [hit for hit in hits if hit.score >= score_threshold]

           ⚠️ 用 ``>=`` 是因为我们用的是 COSINE（**越大越相似**）。
              如果哪天索引换成 L2 距离，这个不等号要反过来，
              而且「Top-K」的取法也要变 —— M5 里已经埋过这个伏笔了。

        5) ``return hits``
        """
        # 1. 校验查询文本
        query = query.strip()
        if not query:
            raise RetrievalError("查询文本不能为空")

        # 2. 校验检索模式
        if mode not in ("dense", "hybrid"):
            raise RetrievalError(f"mode 只能是 'dense' 或 'hybrid'，收到 {mode!r}")

        # 2b. 混合模式不允许传阈值。
        #
        # 两种模式的 score 完全不是一个量纲：
        #   稠密：余弦相似度，0~1
        #   混合：RRF 融合分，约 0~0.033
        # 拿同一个阈值（比如 0.7）去过滤 RRF 分数，会把结果**全部**滤光，
        # 而用户看到的只是「搜不到任何东西」——
        # 这种静默的错误结果，比直接报错难查一百倍。
        # 注意 ``and not rerank``：重排会把分数换成模型给的相关度分（0~1），
        # 量纲重新统一了，这时候设阈值是有意义的。
        if mode == "hybrid" and not rerank and score_threshold is not None:
            raise RetrievalError(
                "混合检索的 score 是 RRF 融合分（约 0~0.033），"
                "和余弦相似度（0~1）不是一个量纲；"
                "用同一套阈值过滤只会得到莫名其妙的结果。"
                "请去掉 score_threshold，或者打开 rerank（重排会把分数换成"
                "模型给的相关度分，量纲就统一了）。"
            )

        # 3. 解析 top_k：只用 None 判断，不能用 or
        k = top_k if top_k is not None else self._s.top_k

        # 3b. 重排器与候选池
        reranker = self._reranker
        if rerank and reranker is None:
            raise RetrievalError(
                "开了 rerank 但没有配置重排器。构造时传入 reranker=create_reranker(settings) 即可。"
            )
        fetch = k
        if rerank:
            fetch = rerank_candidates if rerank_candidates is not None else max(k * 4, 20)

        # 4. 设置总超时，包裹【向量化 + Milvus检索】两段异步逻辑
        timeout = total_timeout if total_timeout is not None else self._s.request_timeout
        try:
            async with asyncio.timeout(timeout):
                vectors = await self._embedder.embed([query])
                if not vectors:
                    raise RetrievalError("Embedding 返回了空结果", source=query[:50])
                if mode == "hybrid":
                    # 两路召回都在 indexer 内部完成，我们只负责给出同一个查询的
                    # 两种表示：向量（稠密那一路）+ 原始文本（BM25 那一路）
                    hits = await self._indexer.hybrid_search(
                        vectors[0], query, top_k=fetch, filter_expr=filter_expr
                    )
                else:
                    hits = await self._indexer.search(
                        vectors[0], top_k=fetch, filter_expr=filter_expr
                    )

                # 精排：注意它在同一个 asyncio.timeout 里 ——
                # 重排也要调远程接口，「整次检索」的截止时间应该把它算进去。
                if rerank and hits and reranker is not None:
                    hits = await self._apply_rerank(reranker, query, hits, top_n=k)
        except TimeoutError as exc:
            raise RetrievalError(f"检索超时（{timeout} 秒）", source=query[:50]) from exc

        # 5. 阈值过滤（走到这里只可能是稠密模式，混合模式在上面被拦住了）
        if score_threshold is not None:
            hits = [hit for hit in hits if hit.score >= score_threshold]

        return hits
