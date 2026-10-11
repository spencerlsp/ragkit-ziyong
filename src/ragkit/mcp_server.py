"""把 ragkit 的检索能力暴露成 MCP 工具，给本地 agent 用。

设计上的五个决定：

1. **只暴露只读检索，不暴露入库/删除。**
   这些工具会被 agent 自动注册给模型 —— 模型能调什么，就等于它能做什么。
   ingest / drop / delete 是运维动作，留在 CLI 里；
   否则模型一句「帮我清理一下」就能把库删了。

2. **标 ``read_only_hint=True``。**
   客户端据此判断要不要人工审批。不标的话，纯读的检索也会每次弹确认，
   在非交互场景（脚本、CI）会被直接拒绝。

3. **返回文本，且不带分数。**
   混合检索的分数是 RRF（0~0.033），加了重排又变成 0~1 —— 同一个字段两种量纲，
   模型解读不了，实测会让它得出「匹配度很低」这种错误结论。
   分数是给调试用的，不是给模型看的。

4. **检索方式由配置决定，工具里没有这个参数。**
   工具签名只有 ``query`` 和 ``top_k``，没有 ``strategy``。
   理由：「用哪种方式检索」是**部署决策** —— 模型既不知道这套库上哪种更好，
   也不知道你到底配没配重排 key。给它一个 strategy 参数，结果就是
   每次随机试一种；更糟的是它会拿着「这次换策略了」的错觉去解读同一批结果 ——
   换一种策略，分数的尺子都变了，同一句话在两个策略下的分数根本不可比。
   要换方式就改配置里的 ``RAGKIT_MCP_STRATEGY``，改一次就一直是它。

5. **用了哪个策略只写日志，不写进返回给模型的文本。**
   这种诊断信息一旦进了上下文，模型很容易把它写进给用户的答案里
   （「我用 hybrid_rerank 检索到……」）。和「不显示分数」是同一条理由。

用法::

    uv sync --extra mcp          # mcp 是可选依赖，默认不装
    uv run python -m ragkit.mcp_server

⚠️ **stdio 模式下 stdout 是 JSON-RPC 协议通道。**
   任何 ``print()``（包括第三方库的调试输出）都会污染协议流，
   让客户端解析失败，而且报错完全指不到原因。
   调试信息一律走 ``logging``（默认输出到 stderr）。

配置（**进程环境变量优先，其次项目根的 .env**，和 CLI 的规则一致）：
    RAGKIT_MCP_STRATEGY   检索策略名，默认 hybrid_rerank。
                          可取的值跑 `ragkit.retrieval.describe_strategies()` 看；
                          **写错会在启动时直接报错**，不会静默降级成别的策略。
                          没配 RERANK_API_KEY 时不要用带 rerank 的策略，
                          否则每次检索都会失败（换成 hybrid 或 dense）。
    RAGKIT_MCP_MODE       ← 已废弃，改用 RAGKIT_MCP_STRATEGY
    RAGKIT_MCP_RERANK     ← 已废弃；仍然认，会按上面的规则映射成策略名并打警告
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import dotenv_values
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .config import Settings
from .errors import RagkitError, RetrievalError
from .pipeline import Ragkit
from .retrieval import DEFAULT_STRATEGY, Strategy, describe_strategies, resolve_strategy
from .schemas import ScoredChunk

logger = logging.getLogger("ragkit.mcp")

# MCP SDK 在导入时会把 root logger 设成 INFO，handler 指向 **stderr**
# （所以不会污染 stdio 的协议通道）。但 httpx 会跟着把每一次请求都打出来，
# 平时纯属噪音 —— 我们自己的「检索完成」那行已经够了，压掉它。
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_TOP_K = 5
MAX_TOP_K = 10  # 模型偶尔会要 100 条，那会把上下文塞爆

# ``env_file=".env"`` 是**相对当前工作目录**解析的，而 MCP server 是被 agent
# 当子进程拉起的 —— 工作目录由客户端决定，通常不是本项目目录。
# 找不到 .env 不会报错，只会静默用默认值（连 localhost 的 Milvus、key 是空串），
# 然后每次检索都失败且看不出原因。所以这里给 .env 一个确定的路径。
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_DOTENV = dotenv_values(_ENV_FILE) if _ENV_FILE.exists() else {}


def _option(name: str, default: str = "") -> str:
    """读一个本模块自己的配置项：**环境变量优先，其次项目根的 `.env`**。

    这两条路径都要认，理由不同：
      * 环境变量 —— MCP 客户端的 ``env`` 块写在这里，它优先级更高，
        等于「这个客户端上怎么跑」；
      * 项目根的 ``.env`` —— 本机改一下最顺手的地方，改完所有客户端都跟着变。

    优先级和 ``Settings`` 保持一致（环境变量 > .env），否则「谁说了算」就得靠猜。
    **只认 os.getenv 是不够的**：那样写进 .env 会静默不生效 ——
    而「配了却没生效」正是这个项目里最难自查的一类问题。
    """
    return os.getenv(name) or str(_DOTENV.get(name) or default)


def _strategy_from_env() -> tuple[str, str | None]:
    """决定这次部署用哪个策略，返回 ``(策略名, 弃用提示或 None)``。

    旧的 ``RAGKIT_MCP_MODE`` / ``RAGKIT_MCP_RERANK`` 仍然认，映射成对应的策略名 ——
    别人的 MCP 客户端配置里写着它们，直接不认就是「升级之后检索悄悄换了招」，
    而且不会有任何提示。**认下来 + 打警告**才是升级该有的样子。
    """
    explicit = _option("RAGKIT_MCP_STRATEGY").strip()
    if explicit:
        return explicit, None

    mode = _option("RAGKIT_MCP_MODE").strip()
    rerank = _option("RAGKIT_MCP_RERANK").strip()
    if not mode and not rerank:
        return DEFAULT_STRATEGY, None

    if (mode or "hybrid") == "dense":
        name = "dense"  # dense + 重排没有对应策略：重排本来就是给混合那一路补精度的
    elif (rerank or "1") not in ("0", "false", "False"):
        name = "hybrid_rerank"
    else:
        name = "hybrid"
    hint = (
        f"RAGKIT_MCP_MODE / RAGKIT_MCP_RERANK 已废弃，本次按 {name} 处理；"
        "请改用 RAGKIT_MCP_STRATEGY"
    )
    return name, hint


_RAW_STRATEGY, _LEGACY_HINT = _strategy_from_env()

# 配置错误要在**进程起来的那一刻**炸，而不是每次调用都失败 ——
# 名字写错（hybird？）的话让他现在就知道，而且消息里把可选项
# 连同「各自擅长什么」一起给全，省得再去翻文档。
try:
    STRATEGY: Strategy = resolve_strategy(_RAW_STRATEGY)
except RetrievalError as exc:
    raise SystemExit(f"{exc}\n{describe_strategies()}") from exc
if _LEGACY_HINT:
    # 这里 logging 还没配 handler，靠 logging 的 lastResort 输出到 stderr。
    # stderr 不参与 stdio 协议，安全。
    logger.warning(_LEGACY_HINT)

mcp = MCPServer("ragkit")

# 长驻进程：Ragkit 只建一次，别每次调用都重建
# （那会把 Milvus 连接和 HTTP 连接池反复重建）。
_rag: Ragkit | None = None
_rag_lock = asyncio.Lock()


async def _client() -> Ragkit:
    """懒加载 + 双检锁。

    为什么要锁：两个请求同时进来会都看到 ``_rag is None``，
    于是建出两个 Ragkit，其中一个的连接池永远不会被关。
    """
    global _rag
    if _rag is not None:
        return _rag
    async with _rag_lock:
        if _rag is None:
            # _env_file 是 pydantic-settings 的运行时参数，不在 Settings 的类型签名里
            settings = Settings(_env_file=_ENV_FILE if _ENV_FILE.exists() else None)  # type: ignore[call-arg]
            _rag = Ragkit(settings)
    return _rag


def _format_hits(hits: list[ScoredChunk]) -> str:
    """把检索结果拼成模型能直接读的文本。

    两条刻意的规则：
      * **空结果要说人话** —— 返回空字符串的话，模型会自己编一段。
      * **不给分数** —— 见模块 docstring 第 3 条。
    """
    if not hits:
        return "没有检索到相关片段。可以换个说法再试一次，或者告诉用户：知识库里可能没有这份资料。"

    blocks: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        # source 是入库时写进 Milvus 动态字段的文件路径，模型靠它标注出处
        source = hit.chunk.metadata.get("source") or hit.chunk.doc_id[:8]
        blocks.append(f"[{rank}] 来源：{source}\n{hit.chunk.text}")
    return "\n\n".join(blocks)


@mcp.tool(
    title="知识库检索",
    annotations=ToolAnnotations(read_only_hint=True),
)
async def search_knowledge_base(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """在知识库里检索相关片段。回答关于已索引文档的问题前先调用它，并在答案里标注来源文件。"""
    # 上限兜一下：模型偶尔会要 100 条，那会把上下文塞爆
    limit = max(1, min(top_k, MAX_TOP_K))

    try:
        rag = await _client()
        # 用哪种检索方式是配置决定的（RAGKIT_MCP_STRATEGY），这里没有开关可拨
        hits = await STRATEGY.search(rag.retriever, query, top_k=limit)
    except RagkitError as exc:
        # 失败要变成**文本**返回，不能抛异常：抛出去客户端只看到
        # "tool call failed"，模型拿不到任何可纠正的信息。
        # 非 RagkitError 的异常（代码 bug）故意不接，让它照常冒出去。
        logger.warning("检索失败（策略 %s）：%s", STRATEGY.name, exc)
        return f"检索失败：{exc}（这是检索服务的问题，不是知识库里没有资料）"

    # 用了哪个策略记进日志就够了，别塞进给模型的文本（见模块 docstring 第 5 条）
    logger.info("检索完成：strategy=%s top_k=%d 命中=%d", STRATEGY.name, limit, len(hits))
    return _format_hits(hits)


def main() -> None:
    try:
        mcp.run()  # 默认 transport="stdio"：由 agent 拉起来，走标准输入输出通信
    finally:
        # ⚠️ 这里**必须**吞掉异常。
        #
        # mcp.run() 内部跑过并关掉了它自己的事件循环，而 Ragkit 的
        # httpx / Milvus 客户端是在那个 loop 上创建的。现在 asyncio.run()
        # 会新建一个 loop，在新 loop 里关旧 loop 的连接池，可能抛
        # "Event loop is closed"。
        #
        # 更要紧的是：这是在退出路径上，抛出去会掩盖真正的异常、污染退出码。
        # 反正进程马上结束，OS 会回收套接字 —— 关不上也不影响任何东西。
        if _rag is not None:
            try:
                asyncio.run(_rag.aclose())
            except Exception:  # noqa: BLE001
                logger.debug("关闭 Ragkit 失败（进程即将退出，忽略）", exc_info=True)


if __name__ == "__main__":
    main()
