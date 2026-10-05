"""MCP 服务端的测试。

``mcp`` 是可选依赖，没装就整份跳过 —— 否则 ``uv sync`` 之后跑测试会直接 ImportError。

这里只测 ``_format_hits``：它是整个 MCP 层里唯一的纯函数，
而且**恰恰是模型直接看到的那段内容** —— 格式错了，模型就会理解错。
工具函数本身（``search_knowledge_base``）要连真 Milvus，属于集成测试的范畴。
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="需要可选依赖：uv sync --extra mcp")

from fakes import make_chunk  # noqa: E402

from ragkit.mcp_server import _format_hits  # noqa: E402
from ragkit.schemas import ScoredChunk  # noqa: E402


def hit(text: str, score: float = 0.9, *, source: str | None = "手册.md") -> ScoredChunk:
    chunk = make_chunk(text).model_copy(update={"metadata": {"source": source} if source else {}})
    return ScoredChunk(chunk=chunk, score=score)


def test_empty_result_tells_a_story() -> None:
    """空结果不能返回空字符串 —— 那样模型会自己编一段。"""
    message = _format_hits([])
    assert message
    assert "没有检索到" in message


def test_single_hit_has_source_and_text() -> None:
    """单条结果要带上出处和正文。"""
    message = _format_hits([hit("第一条内容")])
    assert "手册.md" in message
    assert "第一条内容" in message
    assert message.startswith("[1]")


def test_multiple_hits_are_numbered_in_order() -> None:
    """多条结果按顺序编号，块与块之间空行分隔。"""
    message = _format_hits([hit("甲"), hit("乙"), hit("丙")])
    assert message.index("[1]") < message.index("[2]") < message.index("[3]")
    assert "甲" in message
    assert "丙" in message


def test_missing_source_falls_back_to_doc_id() -> None:
    """source 缺失时回退到 doc_id 的前 8 位，而不是显示 "None"。"""
    message = _format_hits([hit("内容", source=None)])
    assert "None" not in message
    assert "来源：" in message


def test_scores_are_never_exposed() -> None:
    """分数绝不能出现在给模型的文本里。

    RRF 分（0~0.033）和余弦相似度（0~1）量纲不同，
    模型看到一个小数只会得出「匹配度很低」这种错误结论。
    """
    message = _format_hits([hit("内容", score=0.0164)])
    assert "0.0164" not in message
    assert "score" not in message.lower()
