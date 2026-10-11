"""MCP 服务端的测试。

``mcp`` 是可选依赖，没装就整份跳过 —— 否则 ``uv sync`` 之后跑测试会直接 ImportError。

测三类东西，都不需要 Milvus：

  * ``_format_hits`` —— 模型**直接看到的那段内容**，格式错了模型就理解错；
  * 工具签名 —— 检索方式是**部署决策**，模型那边不能有这个参数（钉住它）；
  * 检索方式怎么定 —— 环境变量 > 项目根的 .env > 代码默认，优先级不能乱。

真正要连库的那条路（检索成功、拿到结果）属于集成测试的范畴。
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("mcp", reason="需要可选依赖：uv sync --extra mcp")

from fakes import make_chunk  # noqa: E402

from ragkit.mcp_server import _format_hits, _strategy_from_env, search_knowledge_base  # noqa: E402
from ragkit.retrieval import DEFAULT_STRATEGY  # noqa: E402
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


# ------------------------------------------------- 检索方式：是配置，不是参数


def test_the_tool_does_not_let_the_model_pick_a_strategy() -> None:
    """模型能调什么，就等于它能做什么 —— 所以工具上**不该有**策略参数。

    「用哪种方式检索」是部署决策：模型既不知道这套库上哪种更好，
    也不知道你到底配没配重排 key。给它一个 strategy 参数，结果就是
    每次随机试一种，然后拿着「这次换策略了」的错觉去解读同一批结果 ——
    换一种策略，分数的尺子都变了。

    这是个刻意的设计约束，所以用签名钉住：以后有人「顺手」把它加回来，
    这条测试会立刻变红。
    """
    assert set(inspect.signature(search_knowledge_base).parameters) == {"query", "top_k"}


@pytest.fixture(autouse=True)
def _empty_project_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """把「项目根的 .env」清空。

    ``_DOTENV`` 是模块导入时读的**真实文件** —— 你本机 .env 里要是设了
    RAGKIT_MCP_*，下面这些测试就会随机器而变。测试必须自己控制输入。
    """
    monkeypatch.setattr("ragkit.mcp_server._DOTENV", {})


def test_nothing_configured_means_the_code_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RAGKIT_MCP_STRATEGY", "RAGKIT_MCP_MODE", "RAGKIT_MCP_RERANK"):
        monkeypatch.delenv(key, raising=False)

    assert _strategy_from_env() == (DEFAULT_STRATEGY, None)


def test_env_var_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGKIT_MCP_STRATEGY", "section_aware")

    assert _strategy_from_env() == ("section_aware", None)


def test_project_dotenv_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """写进项目根的 `.env` 也要生效 —— 那才是「本地改一下」最顺手的地方。

    只认 ``os.getenv`` 的话，写进 .env 会**静默不生效**，
    而「配了却没生效」是这个项目里最难自查的一类问题。
    """
    monkeypatch.delenv("RAGKIT_MCP_STRATEGY", raising=False)
    monkeypatch.setattr("ragkit.mcp_server._DOTENV", {"RAGKIT_MCP_STRATEGY": "dense"})

    assert _strategy_from_env() == ("dense", None)


def test_env_var_beats_project_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """两条路都配了时环境变量说了算 —— 和 Settings 的优先级保持一致。"""
    monkeypatch.setenv("RAGKIT_MCP_STRATEGY", "hybrid")
    monkeypatch.setattr("ragkit.mcp_server._DOTENV", {"RAGKIT_MCP_STRATEGY": "dense"})

    assert _strategy_from_env() == ("hybrid", None)


@pytest.mark.parametrize(
    ("mode", "rerank", "expected"),
    [
        ("dense", "1", "dense"),
        ("hybrid", "1", "hybrid_rerank"),
        ("hybrid", "0", "hybrid"),
    ],
)
def test_legacy_env_maps_to_a_strategy(
    monkeypatch: pytest.MonkeyPatch, mode: str, rerank: str, expected: str
) -> None:
    """旧的 MODE/RERANK 仍然认，但要提示换新变量。

    别人的 MCP 客户端配置里写着这两个变量。直接不认，就是
    「升级之后检索悄悄换了招」—— 而且一句话都不会说。
    """
    monkeypatch.delenv("RAGKIT_MCP_STRATEGY", raising=False)
    monkeypatch.setenv("RAGKIT_MCP_MODE", mode)
    monkeypatch.setenv("RAGKIT_MCP_RERANK", rerank)

    name, hint = _strategy_from_env()

    assert name == expected
    assert hint is not None
    assert "RAGKIT_MCP_STRATEGY" in hint
