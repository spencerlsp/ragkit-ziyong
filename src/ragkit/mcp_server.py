"""把 ragkit 的检索能力暴露成 MCP 工具，给本地 agent 用。

设计上的四个决定：

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

4. **检索策略走环境变量，不当工具参数。**
   用 dense 还是 hybrid、要不要重排，是**部署决策**，
   不该由模型每次现挑（它没有判断依据，只会随便试）。

用法::

    uv sync --extra mcp          # mcp 是可选依赖，默认不装
    uv run python -m ragkit.mcp_server

⚠️ **stdio 模式下 stdout 是 JSON-RPC 协议通道。**
   任何 ``print()``（包括第三方库的调试输出）都会污染协议流，
   让客户端解析失败，而且报错完全指不到原因。
   调试信息一律走 ``logging``（默认输出到 stderr）。

环境变量：
    RAGKIT_MCP_MODE     dense | hybrid，默认 hybrid
    RAGKIT_MCP_RERANK   0 / false 关闭重排，默认开启
                        （没配 RERANK_API_KEY 的话必须设成 0，否则每次检索都会失败）
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Literal, cast

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .config import Settings
from .errors import RagkitError
from .pipeline import Ragkit
from .schemas import ScoredChunk

logger = logging.getLogger("ragkit.mcp")

Mode = Literal["dense", "hybrid"]

# os.getenv 返回 str，但下面会校验取值；cast 只是把「这里刚校验过」这件事告诉 mypy
MODE: Mode = cast(Mode, os.getenv("RAGKIT_MCP_MODE", "hybrid"))
USE_RERANK = os.getenv("RAGKIT_MCP_RERANK", "1") not in ("0", "false", "False")

DEFAULT_TOP_K = 5
MAX_TOP_K = 10  # 模型偶尔会要 100 条，那会把上下文塞爆

# 配置错误要在**进程起来的那一刻**炸，而不是每次调用都失败。
# 用户设了 RAGKIT_MCP_MODE=hybird（拼错）的话，让他现在就知道。
if MODE not in ("dense", "hybrid"):
    raise SystemExit(f"RAGKIT_MCP_MODE 只能是 'dense' 或 'hybrid'，收到 {MODE!r}")

# ``env_file=".env"`` 是**相对当前工作目录**解析的，而 MCP server 是被 agent
# 当子进程拉起的 —— 工作目录由客户端决定，通常不是本项目目录。
# 找不到 .env 不会报错，只会静默用默认值（连 localhost 的 Milvus、key 是空串），
# 然后每次检索都失败且看不出原因。所以这里给 .env 一个确定的路径。
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

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
    limit = max(1, min(top_k, MAX_TOP_K))

    try:
        rag = await _client()
        hits = await rag.query(query, top_k=limit, mode=MODE, rerank=USE_RERANK)
    except RagkitError as exc:
        # 失败要变成**文本**返回，不能抛异常：抛出去客户端只看到
        # "tool call failed"，模型拿不到任何可纠正的信息。
        # 非 RagkitError 的异常（代码 bug）故意不接，让它照常冒出去。
        logger.warning("检索失败：%s", exc)
        return f"检索失败：{exc}（这是检索服务的问题，不是知识库里没有资料）"

    return _format_hits(hits)


def main() -> None:
    try:
        mcp.run()  # 默认 transport="stdio"：由 agent 拉起来，走标准输入输出通信
    finally:
        # 进程退出时 OS 会关掉套接字，所以这是「优雅」而非「必须」；
        # 但有些客户端会反复重启 server 进程，别每次都留一个没关的连接池。
        if _rag is not None:
            asyncio.run(_rag.aclose())


if __name__ == "__main__":
    main()
