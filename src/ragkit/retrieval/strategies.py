"""有名字的检索策略：把「哪一层检索器 + 什么模式 + 要不要重排」收成一份清单。

为什么要给组合起名字：
    稠密、混合、重排、结构感知后处理是分四次加进来的，每加一次组合数就翻一倍。
    不起名字的话，每个消费方（MCP 服务端、CLI、评估脚本、你自己的脚本）
    都得自己拼一遍参数 —— 拼错一个就得到另一套结果，而且**不报错**。
    起名字之后，「换一种检索方式」在调用方眼里就是换一个字符串。

新增一种检索方式 = 往 ``STRATEGIES`` 里加一条，要做的事只有三件：
    1. 想清楚它对应哪个 ``mode``、要不要 ``rerank``；
    2. 需要后处理层的话，写一个 ``wrap(base) -> retriever`` 的小函数；
    3. 给一句 ``summary`` —— 这句话会被**写进 MCP 的工具说明**，
       模型是照着它挑策略的，所以要写「它擅长什么」而不是「它是什么」。

和 CLI 的对应关系（两边表达的是同一件事，换名字时记得一起改）：

    dense                ~  --mode dense
    hybrid               ~  --mode hybrid
    hybrid_rerank        ~  --mode hybrid --rerank
    section_aware        ~  --mode hybrid --section-aware
    section_aware_rerank ~  --mode hybrid --section-aware --rerank
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..errors import RetrievalError
from ..schemas import ScoredChunk
from .retriever import Retriever
from .section_aware import SectionAwareRetriever

__all__ = [
    "Mode",
    "Strategy",
    "STRATEGIES",
    "DEFAULT_STRATEGY",
    "resolve_strategy",
    "describe_strategies",
]

Mode = Literal["dense", "hybrid"]

# wrap 的入参永远是「通用检索器」，出参是「能 retrieve 的东西」。
Wrap = Callable[[Retriever], Retriever | SectionAwareRetriever]


def _plain(base: Retriever) -> Retriever:
    """不加后处理层。"""
    return base


def _section_aware(base: Retriever) -> SectionAwareRetriever:
    """套一层结构感知后处理。

    参数一律走 ``SectionAwareRetriever`` 自己的默认值 —— 在这里再抄一遍
    （比如 4 / 0.5 / 2）就等于埋下一处迟早会漂移的副本。
    """
    return SectionAwareRetriever(base=base)


@dataclass(frozen=True, slots=True)
class Strategy:
    """一种检索方式。``name`` 是它对外的唯一标识（也就是配置里写的那个字符串）。"""

    name: str
    summary: str
    mode: Mode
    rerank: bool
    wrap: Wrap = _plain

    async def search(
        self, base: Retriever, query: str, *, top_k: int | None = None
    ) -> list[ScoredChunk]:
        """用这套策略跑一次检索，返回带分数的片段。

        ``base`` 是**通用检索器**（``Ragkit.retriever``），生命周期归调用方；
        这里只按需包一层后处理。后处理层不持有任何资源（没有连接、没有线程），
        所以每次现造现丢，不需要缓存。
        """
        retriever = self.wrap(base)
        return await retriever.retrieve(query, top_k=top_k, mode=self.mode, rerank=self.rerank)


# 名字是**对外的接口**：会写进 .env（RAGKIT_MCP_STRATEGY），也会出现在工具说明里。
# 改名等于破坏兼容，加名字才是安全的。
STRATEGIES: dict[str, Strategy] = {
    strategy.name: strategy
    for strategy in (
        Strategy(
            name="dense",
            summary="纯向量检索。最快最省，擅长「换一种说法问同一件事」",
            mode="dense",
            rerank=False,
        ),
        Strategy(
            name="hybrid",
            summary="向量 + BM25 关键词（RRF 融合）。擅长含专有名词、编号、罕见词的提问",
            mode="hybrid",
            rerank=False,
        ),
        Strategy(
            name="hybrid_rerank",
            summary="hybrid 再加重排。最准也最慢最贵，擅长「提问用词和原文差得很远」",
            mode="hybrid",
            rerank=True,
        ),
        Strategy(
            name="section_aware",
            summary=(
                "hybrid + 结构感知后处理（标题加权 / 去近似重复 / 同节限流）。"
                "要求文档入库时用了 --splitter section，否则没有标题可加权"
            ),
            mode="hybrid",
            rerank=False,
            wrap=_section_aware,
        ),
        Strategy(
            name="section_aware_rerank",
            summary="section_aware 再加重排。工程文档语料上通常是最强的一档",
            mode="hybrid",
            rerank=True,
            wrap=_section_aware,
        ),
    )
}

DEFAULT_STRATEGY = "hybrid_rerank"


def resolve_strategy(name: str) -> Strategy:
    """按名字取策略；名字不认识就报错，并把合法的名字**全列出来**。

    错误信息里列清单是刻意的：调用方（尤其是模型）拿到这一句就能自己纠对，
    不用去翻文档 —— 而这正是「把失败变成一句可纠正的话」的用法。
    """
    key = (name or "").strip()
    if key not in STRATEGIES:
        raise RetrievalError(f"未知的检索策略 {name!r}；可用的是：{', '.join(STRATEGIES)}")
    return STRATEGIES[key]


def describe_strategies(*, default: str = DEFAULT_STRATEGY) -> str:
    """一行一个策略的清单，``*`` 标出哪个是默认。

    ``default`` 由调用方给：MCP 服务端的默认不一定等于代码里的默认
    （部署可以用 ``RAGKIT_MCP_STRATEGY`` 改），标错了模型就会挑错。
    """
    lines = [
        f"{'*' if name == default else ' '} {name}：{strategy.summary}"
        for name, strategy in STRATEGIES.items()
    ]
    return f"可用检索策略（* 为默认 {default}）：\n" + "\n".join(lines)
