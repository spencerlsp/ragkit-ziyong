"""结构感知的检索：在通用检索之上，按文档结构收敛结果。

为什么需要它（针对「同一主题的工程文档」这个场景）：

    * **摘要层会抢答**。像 LangChain 这种文档库，``index.md`` 会给每篇文档写一句
      摘要，每个文件 frontmatter 里又有自己的 ``description``。
      同一句话在语料里出现好几次，向量检索特别容易把「摘要」排到
      「真正讲这件事的那一节」前面 —— 摘要确实相关，但它给不出细节。
    * **通用检索不知道标题**。命中的一段代码属于哪一节、讲的是 middleware
      还是 callbacks，查询里出现这些词时本该加权，
      但向量和 BM25 都把这一行当成普通正文。
    * **top_k 会被同一节吃光**。一节被切成 5 块、查询又正好命中这一节，
      返回的前 5 条可能全是同一节的不同片段，其他相关内容一条都进不来。

这个检索器做三件事（**都在通用检索之后**做，不碰 Milvus、不加模型调用）：

    1. **扩召回**：先取 ``candidate_factor`` 倍的候选，给后处理留挑选空间 ——
       只在 k 条里重排，等于白干。
    2. **标题加权**：拿查询词和 chunk 开头的 ``[doc: ... | section: ...]`` 比一比，
       重合越多加得越多。用**乘法**加成，因为 score 的量纲随模式而变
       （余弦 0~1 / RRF 约 0.03 / 重排 0~1），乘法对量纲不敏感，加法会直接算错。
    3. **去近似重复 + 抑制同节刷屏**：先扔掉文本高度重叠的块，
       再限制同一节最多贡献几块。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from ..config import get_settings
from ..errors import RetrievalError
from ..schemas import ScoredChunk
from .retriever import Retriever

__all__ = ["SectionAwareRetriever"]

# 切分器补在每块开头的那一行：[doc: X | section: A > B]
_CONTEXT_RE = re.compile(
    r"^\[doc:\s*(?P<doc>[^|\]]+?)\s*(?:\|\s*section:\s*(?P<section>[^\]]*?)\s*)?\]"
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# 这些词几乎出现在每个 chunk 的前缀里，拿它们做标题匹配只会引入噪声
_STOPWORDS = frozenset(
    {
        "doc",
        "section",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "how",
        "what",
        "when",
        "which",
        "where",
        "does",
        "into",
        "not",
        "but",
        "its",
        "their",
        "you",
        "your",
        "can",
        "use",
        "used",
    }
)


def _tokens(text: str) -> set[str]:
    """粗略分词：英文按字母数字串切。够用，且不引入任何依赖。"""
    return {
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if len(token) > 2 and token not in _STOPWORDS
    }


def _shingles(text: str, size: int = 4) -> frozenset[str]:
    """按词做 4-gram，用来判断两块是不是「近似重复」。"""
    words = [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
    if len(words) < size:
        return frozenset(words)
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


class SectionAwareRetriever:
    """包一层通用检索器，按文档结构做后处理的检索器。"""

    def __init__(
        self,
        *,
        base: Retriever | None = None,
        candidate_factor: int = 4,
        section_boost: float = 0.5,
        dedupe_threshold: float = 0.8,
        max_per_section: int | None = 2,
        **base_kwargs: object,
    ) -> None:
        """``base`` 为空时，用 ``base_kwargs`` 现造一个通用 ``Retriever``。

        ``section_boost``
            标题重合带来的最大加成比例。0.5 表示标题全中时分数最多乘 1.5。
        ``dedupe_threshold``
            两块的词 4-gram 相似度超过它就认为在讲同一件事，只留高分那条。
        ``max_per_section``
            同一节最多返回几块；``None`` 表示不限制。
        """
        if candidate_factor < 1:
            raise RetrievalError(f"candidate_factor 必须 >= 1，收到 {candidate_factor}")
        if section_boost < 0:
            raise RetrievalError(f"section_boost 不能为负数，收到 {section_boost}")
        if not 0.0 <= dedupe_threshold <= 1.0:
            raise RetrievalError(f"dedupe_threshold 必须落在 [0, 1]，收到 {dedupe_threshold}")
        if max_per_section is not None and max_per_section < 1:
            raise RetrievalError(f"max_per_section 必须 >= 1，收到 {max_per_section}")

        self._owns_base = base is None
        self._base = base or Retriever(**base_kwargs)  # type: ignore[arg-type]
        self.candidate_factor = candidate_factor
        self.section_boost = section_boost
        self.dedupe_threshold = dedupe_threshold
        self.max_per_section = max_per_section

    async def __aenter__(self) -> SectionAwareRetriever:
        """支持 ``async with SectionAwareRetriever(...) as r:``。"""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """只关自己创建的那层；外部注入的 ``base`` 由调用方负责。"""
        if self._owns_base:
            await self._base.aclose()

    @property
    def base(self) -> Retriever:
        """暴露底层检索器 —— 想直接对比「原始召回」时要能拿到它。"""
        return self._base

    # ------------------------------------------------------------------ 主流程

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: Literal["dense", "hybrid"] = "dense",
        rerank: bool = False,
        rerank_candidates: int | None = None,
        score_threshold: float | None = None,
        filter_expr: str = "",
        total_timeout: float | None = None,
    ) -> list[ScoredChunk]:
        """检索并做结构后处理。参数含义与 :meth:`Retriever.retrieve` 完全一致。"""
        query = query.strip()
        if not query:
            raise RetrievalError("查询文本不能为空")

        k = top_k if top_k is not None else get_settings().top_k
        fetch = k * self.candidate_factor

        hits = await self._base.retrieve(
            query,
            top_k=fetch,
            mode=mode,
            rerank=rerank,
            rerank_candidates=rerank_candidates,
            filter_expr=filter_expr,
            total_timeout=total_timeout,
        )

        ranked = self._boost_by_section(query, hits)
        ranked = self._dedupe(ranked)
        ranked = self._limit_per_section(ranked)

        if score_threshold is not None:
            # 阈值仍按底层分数判断 —— 与 Retriever.retrieve 的约定保持一致：
            # 只有稠密模式和重排模式的 score 是「越大越相关且落在 0~1」。
            ranked = [hit for hit in ranked if hit.score >= score_threshold]

        return ranked[:k]

    # ------------------------------------------------------------------ 后处理

    @staticmethod
    def _context_of(text: str) -> tuple[str, str]:
        """从 chunk 开头那行取出 ``(文档标题, 标题路径)``；没有就返回空串。"""
        match = _CONTEXT_RE.match(text.lstrip())
        if match is None:
            return "", ""
        return (match.group("doc") or "").strip(), (match.group("section") or "").strip()

    def _boost_by_section(self, query: str, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """标题路径命中查询词的块，分数按重合比例上浮。"""
        query_tokens = _tokens(query)
        if not query_tokens or self.section_boost <= 0:
            return list(hits)

        boosted: list[ScoredChunk] = []
        for hit in hits:
            doc_title, section = self._context_of(hit.chunk.text)
            heading_tokens = _tokens(f"{doc_title} {section}")
            overlap = len(query_tokens & heading_tokens) / len(query_tokens)
            boosted.append(
                ScoredChunk(
                    chunk=hit.chunk,
                    score=hit.score * (1.0 + self.section_boost * overlap),
                )
            )
        boosted.sort(key=lambda item: item.score, reverse=True)
        return boosted

    def _dedupe(self, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """讲同一件事的块只留高分那条（摘要 vs 正文本体的经典重复）。

        比的是**去掉前缀的正文**：``[doc | section]`` 那一行每块都不一样，
        把它算进去只会稀释相似度，让重复的两块看起来「没那么像」。
        """
        kept: list[ScoredChunk] = []
        kept_shingles: list[frozenset[str]] = []
        for hit in hits:
            current = _shingles(self._body_of(hit.chunk.text))
            if any(
                self._similar(current, other) >= self.dedupe_threshold for other in kept_shingles
            ):
                continue
            kept.append(hit)
            kept_shingles.append(current)
        return kept

    @staticmethod
    def _body_of(text: str) -> str:
        """剥掉切分器补的 ``[doc | section]`` 前缀行，只留正文。"""
        stripped = text.lstrip()
        match = _CONTEXT_RE.match(stripped)
        if match is None:
            return text
        return stripped[match.end() :].lstrip("\n")

    @staticmethod
    def _similar(left: frozenset[str], right: frozenset[str]) -> float:
        """重叠系数（overlap coefficient）= 交集 / 较小的一边。

        **这里刻意不用 Jaccard**。这份语料里最常见的一种重复是
        「一句话摘要被整段正文包含」：

            summary: "... middleware hooks wrap model calls and tool calls"
            detail : "... 同样这句话 ... 后面还接着讲怎么改 payload"

        摘要的 4-gram 几乎全在 detail 里，但 detail 还多出一大截 ——
        用 Jaccard，分母（并集）被拉大，算出来只有 0.2，去重完全失效；
        用重叠系数则是 1.0，一击命中。

        代价是它对「长短悬殊」更敏感，所以阈值不能调低：
        切分重叠造成的相邻块通常只有 15% 左右的重合，远够不到 0.8。
        """
        if not left or not right:
            return 0.0
        return len(left & right) / min(len(left), len(right))

    def _limit_per_section(self, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """同一节最多留 ``max_per_section`` 块，防止一节刷屏挤掉其他内容。"""
        if self.max_per_section is None:
            return list(hits)

        counts: dict[tuple[str, str], int] = {}
        out: list[ScoredChunk] = []
        for hit in hits:
            _, section = self._context_of(hit.chunk.text)
            key = (hit.chunk.doc_id, section)
            if counts.get(key, 0) >= self.max_per_section:
                continue
            counts[key] = counts.get(key, 0) + 1
            out.append(hit)
        return out
