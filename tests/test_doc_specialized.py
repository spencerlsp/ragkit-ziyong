"""工程文档专属切分器 / 检索器的离线测试。

这两件东西都是为「同一主题的工程文档」这个场景定制的：

    MarkdownSectionSplitter   按标题分节、保护代码块、剥掉 frontmatter
    SectionAwareRetriever     标题加权、去近似重复、抑制同节刷屏

测试就按这几件事分组写。**全部离线** —— 不连 Milvus、不调任何 API，
和项目里其它测试保持一个规矩。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from fakes import StubEmbedder, StubStore

from ragkit.errors import RetrievalError, SplitError
from ragkit.retrieval import Retriever, SectionAwareRetriever
from ragkit.schemas import Chunk, Document, ScoredChunk
from ragkit.splitting import MarkdownSectionSplitter
from ragkit.utils import stable_id

DOC = """---
title: "LangChain System Architecture"
sources:
  - id: s1
    resource: repo://libs/core/foo.py
---

# LangChain System Architecture

## Overview

这一节介绍三层架构。

## Streaming

流式输出按 token 返回。
"""


def make_document(text: str = DOC, doc_id: str = "d1") -> Document:
    """造一个带 frontmatter 的工程文档。"""
    return Document(doc_id=doc_id, text=text, source="architecture.md")


def make_hit(text: str, score: float, *, doc_id: str = "d1", index: int = 0) -> ScoredChunk:
    """造一条检索结果（chunk_id 用内容哈希，和真实链路一致）。"""
    chunk = Chunk(
        chunk_id=stable_id(doc_id, str(index), text),
        doc_id=doc_id,
        text=text,
        index=index,
        metadata={"source": f"{doc_id}.md"},
    )
    return ScoredChunk(chunk=chunk, score=score)


# ------------------------------------------------------------------ 切分器


def test_frontmatter_is_stripped_and_title_survives() -> None:
    """frontmatter 是元数据不是正文：剥掉它，但把 title 留下来当文档标题。"""
    splitter = MarkdownSectionSplitter(chunk_size=400, merge_small_sections=False)
    chunks = splitter.split(make_document())
    joined = "\n".join(chunk.text for chunk in chunks)

    assert "repo://" not in joined
    assert "sources:" not in joined
    assert chunks[0].metadata["doc_title"] == "LangChain System Architecture"
    assert all(chunk.metadata["splitter"] == "markdown_section" for chunk in chunks)


def test_context_line_carries_heading_path() -> None:
    """每块开头要带 ``[doc | section]``，而且嵌套标题要拼成完整路径。"""
    doc = make_document("# Title\n\n## Middleware\n\n### Hooks\n\n这里讲 hook 的类型。\n")
    chunks = MarkdownSectionSplitter(chunk_size=400, merge_small_sections=False).split(doc)
    sections = [chunk.metadata["section"] for chunk in chunks]

    assert "Middleware > Hooks" in sections
    assert any(
        chunk.text.startswith("[doc: Title | section: Middleware > Hooks]") for chunk in chunks
    )


def test_context_line_can_be_turned_off() -> None:
    """关掉上下文就是消融实验的对照组：不补标题，其余行为不变。"""
    doc = make_document("# Title\n\n## Streaming\n\n按 token 返回。\n")
    chunks = MarkdownSectionSplitter(chunk_size=400, include_context=False).split(doc)

    assert all(not chunk.text.startswith("[doc:") for chunk in chunks)


def test_code_fence_is_never_split() -> None:
    """超长代码块宁可整块超长，也不能切出半截代码。"""
    body = "## Sample\n\n```python\n" + "\n".join(f"print({i})" for i in range(80)) + "\n```\n"
    chunks = MarkdownSectionSplitter(chunk_size=200).split(make_document(body))

    assert len(chunks) == 1
    assert chunks[0].text.count("```") == 2


def test_heading_inside_code_fence_is_not_a_heading() -> None:
    """代码块里的 ``# 注释`` 不能把文档结构带偏。"""
    body = "# Title\n\n## Real Section\n\n```bash\n# 这是注释不是标题\n## 也不是\n```\n"
    chunks = MarkdownSectionSplitter(chunk_size=800).split(make_document(body))

    assert [chunk.metadata["section"] for chunk in chunks] == ["Real Section"]


def test_small_sibling_sections_are_merged() -> None:
    """相邻的小兄弟节并成一块；不并的话会切出一堆信息量极低的碎片。"""
    body = "# T\n\n## Alpha\n\n很短的一节。\n\n## Beta\n\n也很短。\n"

    separate = MarkdownSectionSplitter(chunk_size=800, merge_small_sections=False).split(
        make_document(body)
    )
    merged = MarkdownSectionSplitter(chunk_size=800).split(make_document(body))

    assert len(separate) == 2
    assert len(merged) == 1
    # 合并不能把标题弄丢：两个小标题都要还在正文里
    assert "## Alpha" in merged[0].text
    assert "## Beta" in merged[0].text


def test_chunk_ids_are_stable_and_indexed() -> None:
    """同一个文档切两次，chunk_id 必须一模一样（否则重复入库会堆数据）。"""
    splitter = MarkdownSectionSplitter(chunk_size=400)
    first = splitter.split(make_document())
    second = splitter.split(make_document())

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.index for chunk in first] == list(range(len(first)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": 100, "chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
    ],
)
def test_splitter_rejects_bad_params(kwargs: dict[str, int]) -> None:
    """参数校验要和另外两个切分器保持一致，别让非法参数走到切分中途才炸。"""
    with pytest.raises(SplitError):
        MarkdownSectionSplitter(**kwargs)


# ------------------------------------------------------------------ 检索器


async def _stub_retriever(hits: Sequence[ScoredChunk], store: StubStore | None = None) -> Retriever:
    return Retriever(embedder=StubEmbedder(), indexer=store or StubStore(list(hits)))


async def test_section_boost_promotes_heading_match() -> None:
    """分数相同时，标题路径命中查询的那块应该排到前面。"""
    callbacks = make_hit("[doc: LangChain | section: Callbacks]\n\n回调用来观测执行过程。", 0.50)
    middleware = make_hit(
        "[doc: LangChain | section: Middleware Hooks]\n\n中间件在模型调用前后处理请求。", 0.50
    )

    async with await _stub_retriever([callbacks, middleware]) as base:
        retriever = SectionAwareRetriever(base=base, candidate_factor=1)
        out = await retriever.retrieve("how do middleware hooks work", top_k=2)

    assert out[0].chunk.chunk_id == middleware.chunk.chunk_id


async def test_dedupe_drops_summary_that_body_contains() -> None:
    """摘要被正文整段包含时只留高分那条。

    这是这份语料的真实重复形态：``index.md`` 的一句话摘要，和每个文件
    frontmatter 里的 ``description`` —— 往往是逐字相同的两段。
    注意这里用的是「包含」而不是「Jaccard 高」：
    摘要全在正文里，但正文比摘要长得多，Jaccard 会被并集稀释掉。
    """
    summary = make_hit(
        "[doc: X | section: Summary]\n\nMiddleware hooks wrap model calls and tool calls.",
        0.9,
    )
    detail = make_hit(
        "[doc: X | section: Middleware Hooks]\n\n"
        "Middleware hooks wrap model calls and tool calls. "
        "They also let you rewrite the request payload before it is sent, "
        "which is how retries and caching get implemented.",
        0.8,
    )

    async with await _stub_retriever([summary, detail]) as base:
        retriever = SectionAwareRetriever(base=base, candidate_factor=1, section_boost=0.0)
        out = await retriever.retrieve("middleware hooks", top_k=2)

    assert len(out) == 1
    assert out[0].chunk.chunk_id == summary.chunk.chunk_id


async def test_dedupe_keeps_distinct_chunks() -> None:
    """去重别误伤：两块只是同主题、内容不同，必须都留下。"""
    first = make_hit(
        "[doc: X | section: Tools]\n\nTools are schema-aware callables the model can invoke.",
        0.9,
    )
    second = make_hit(
        "[doc: X | section: Callbacks]\n\nCallbacks observe execution events such as retries.",
        0.8,
    )

    async with await _stub_retriever([first, second]) as base:
        retriever = SectionAwareRetriever(base=base, candidate_factor=1, section_boost=0.0)
        out = await retriever.retrieve("anything", top_k=2)

    assert len(out) == 2


async def test_max_per_section_stops_one_section_flooding() -> None:
    """同一节最多贡献固定条数，否则它会吃掉整个 top_k。"""
    hits = [
        make_hit(
            "[doc: X | section: Same]\n\n" + " ".join(f"token{i}x{j}" for j in range(40)),
            0.9 - i * 0.01,
        )
        for i in range(4)
    ]

    async with await _stub_retriever(hits) as base:
        retriever = SectionAwareRetriever(
            base=base, candidate_factor=1, section_boost=0.0, max_per_section=2
        )
        out = await retriever.retrieve("anything", top_k=4)

    assert len(out) == 2


async def test_candidate_pool_is_widened() -> None:
    """扩召回：传给底层检索库的 top_k 必须放大，后处理才有得挑。"""

    class RecordingStore(StubStore):
        def __init__(self, hits: list[ScoredChunk]) -> None:
            super().__init__(hits)
            self.top_ks: list[int] = []

        async def search(
            self, query_vector: Sequence[float], *, top_k: int = 5, filter_expr: str = ""
        ) -> list[ScoredChunk]:
            self.top_ks.append(top_k)
            return await super().search(query_vector, top_k=top_k, filter_expr=filter_expr)

    hits = [
        make_hit(f"[doc: X | section: S{i}]\n\n" + " ".join(f"w{i}z{j}" for j in range(30)), 0.5)
        for i in range(8)
    ]
    store = RecordingStore(hits)

    async with await _stub_retriever(hits, store) as base:
        retriever = SectionAwareRetriever(
            base=base, candidate_factor=4, section_boost=0.0, dedupe_threshold=1.0
        )
        await retriever.retrieve("anything", top_k=2)

    assert store.top_ks == [8]


async def test_section_aware_returns_at_most_top_k() -> None:
    """后处理之后再截断：保证返回条数不超过调用方要的 k。"""
    hits = [
        make_hit(f"[doc: X | section: S{i}]\n\n" + " ".join(f"q{i}r{j}" for j in range(30)), 0.5)
        for i in range(10)
    ]

    async with await _stub_retriever(hits) as base:
        retriever = SectionAwareRetriever(
            base=base, candidate_factor=3, section_boost=0.0, max_per_section=None
        )
        out = await retriever.retrieve("anything", top_k=3)

    assert len(out) == 3


async def test_empty_query_is_rejected() -> None:
    async with await _stub_retriever([]) as base:
        retriever = SectionAwareRetriever(base=base)
        with pytest.raises(RetrievalError):
            await retriever.retrieve("   ")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_factor": 0},
        {"section_boost": -0.1},
        {"dedupe_threshold": 1.5},
        {"max_per_section": 0},
    ],
)
def test_retriever_rejects_bad_params(kwargs: dict[str, float]) -> None:
    """非法参数在构造时就报错，别等到检索中途才出现莫名其妙的结果。"""
    with pytest.raises(RetrievalError):
        SectionAwareRetriever(**kwargs)  # type: ignore[arg-type]


async def test_externally_injected_base_is_not_closed() -> None:
    """外部注入的底层检索器不归我们关 —— 「谁创建谁销毁」这条规矩不能破。"""
    async with await _stub_retriever([]) as base:
        retriever = SectionAwareRetriever(base=base)
        await retriever.aclose()
        assert base._embedder.closed is False  # noqa: SLF001
