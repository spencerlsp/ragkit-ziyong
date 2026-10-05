"""ragkit 的命令行入口。

分工：**写的操作走 CLI，读的操作通过 MCP 暴露**。
入库、切分、评测由人显式触发；检索作为只读工具给 agent 随时调用。

但 ``query`` 仍然留在 CLI 里 —— 调试时你要能脱离 agent 手动跑一次。
「模型说搜不到」的时候，第一件事是自己在终端敲一遍，看是检索真的坏了，
还是模型用错了。

用法::

    uv run ragkit --help
    uv run ragkit split docs/手册.md --out chunks.jsonl
    uv run ragkit ingest docs/ --reset
    uv run ragkit query "住宿费上限" --mode hybrid --rerank

退出码：``0`` 成功、``1`` 业务失败（RagkitError）、``2`` 参数错误。
**别小看这个区分** —— 很多 CLI 一律 exit(1)，于是脚本里分不清
「参数写错了」和「检索失败了」。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import get_settings
from .errors import RagkitError
from .evaluation import EvalReport, evaluate, load_dataset
from .parsing import parse_file, parse_files
from .pipeline import Ragkit
from .processing import clean_document
from .schemas import ScoredChunk
from .splitting import RecursiveSplitter, write_chunks_jsonl

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


def _emit_json(payload: Any) -> None:
    """机器可读的输出。``ensure_ascii=False`` 让中文直接可读而不是 \\uXXXX。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _show(hit: ScoredChunk, rank: int) -> str:
    source = hit.chunk.metadata.get("source") or hit.chunk.doc_id[:8]
    preview = " ".join(hit.chunk.text.split())[:70]
    return f"{rank:>2}. {hit.score:>7.4f}  {source}\n    {preview}…"


def _leaf_errors(error: BaseException) -> list[BaseException]:
    """把 ExceptionGroup 拆成最内层的异常（可能嵌套）。

    批量解析用的是 TaskGroup，子任务的异常会被打包成 ExceptionGroup（PEP 654）。
    对命令行来说，用户要的是一句人话，不是「unhandled errors in a TaskGroup」。
    """
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _leaf_errors(sub)]
    return [error]


def _eval_mismatch_hint(report: EvalReport) -> str | None:
    """命中率为 0、却没有任何样本失败时，几乎一定是标注和库里的数据对不上。

    这是这个项目里最容易踩、也最难自查的坑：``chunk_id`` 绑定在切分参数上，
    改了 ``CHUNK_SIZE`` / ``CHUNK_OVERLAP`` 就会全变；而检索本身**完全正常**，
    只是拿到的 id 和你标注的 id 一个都对不上 —— 指标于是全是 0，一句报错都没有。

    错误是静默的，所以这里必须替用户问一句「你确定两边是同一次切分吗」。
    """
    if report.n_failed or not report.results:
        return None
    if report.metrics.get("hit_rate", 0.0) > 0.0:
        return None
    return (
        "命中率是 0，但没有任何样本失败 —— 这通常不是检索坏了，"
        "而是**标注和库里的数据对不上**。\n"
        "  最常见的原因：chunk_id 绑定在切分参数上，"
        "入库时用的参数和生成标注时不一样。\n"
        "  检查：先 `ragkit count` 看库里有多少条；确认 .env 里的"
        " CHUNK_SIZE / CHUNK_OVERLAP 和生成标注时一致，"
        "或者重新 `ragkit ingest <文档> --reset` 后按当前参数重新标注。"
    )


# ---------------------------------------------------------------------------
# 只读、不碰数据库的命令
# ---------------------------------------------------------------------------


async def cmd_parse(args: argparse.Namespace) -> int:
    """解析文件并报告字数，不入库 —— 用来确认「解析这一层」是否正常。"""
    documents = await parse_files([Path(p) for p in args.paths])

    if args.json:
        _emit_json(
            [{"source": d.source, "chars": d.char_count, "doc_id": d.doc_id} for d in documents]
        )
        return EXIT_OK

    for document in documents:
        print(f"{document.char_count:>8} 字  {document.source}")
    total = sum(d.char_count for d in documents)
    print(f"\n{len(documents)} 个文件，共 {total} 字")
    return EXIT_OK


async def cmd_split(args: argparse.Namespace) -> int:
    """切分单个文件。``--out`` 导出 JSONL，方便肉眼检查切得好不好。

    这一步**完全不碰数据库** —— 调切分参数时最常用的动作，
    不该被 Milvus 或 API key 卡住。
    """
    settings = get_settings()
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap

    if args.chunk_overlap is None and settings.chunk_overlap >= chunk_size:
        # 用户只调了 chunk-size，重叠还按配置来 —— 但配置值放不下了。
        # 这里**不偷偷改参数**（那会让"我设了 80 怎么切出来不一样"变成谜），
        # 只给一条能直接照做的提示。
        print(
            f"提示：CHUNK_OVERLAP={settings.chunk_overlap} 不小于 chunk_size={chunk_size}，"
            f"需要同时给 --chunk-overlap（建议 {chunk_size // 5} 左右）",
            file=sys.stderr,
        )

    document = clean_document(await parse_file(args.path))
    chunks = RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split(document)

    if args.out:
        written = await write_chunks_jsonl(chunks, Path(args.out))
        print(f"{written} 块 -> {args.out}")

    if args.json:
        _emit_json(
            {
                "doc_id": document.doc_id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunks": [
                    {"index": c.index, "chars": c.char_count, "chunk_id": c.chunk_id}
                    for c in chunks
                ],
            }
        )
        return EXIT_OK

    sizes = [c.char_count for c in chunks] or [0]
    print(
        f"{args.path}  ->  {len(chunks)} 块  "
        f"（min={min(sizes)} max={max(sizes)} 平均={sum(sizes) // len(sizes)}，"
        f"chunk_size={chunk_size} overlap={chunk_overlap}）"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# 碰数据库的命令
# ---------------------------------------------------------------------------


async def cmd_ingest(args: argparse.Namespace) -> int:
    async with Ragkit() as rag:
        result = await rag.ingest(args.paths, reset=args.reset, flush=not args.no_flush)

    if args.json:
        _emit_json(result.model_dump())
        return EXIT_OK

    print(f"入库完成：{result.files} 个文件  {result.chunks} 个 chunk  {result.written} 条记录")
    if result.chunks != result.written:
        # 这三个数本该相等，不等就说明中间某一步吞了东西
        print("⚠️ chunk 数和写入数不一致，检查一下", file=sys.stderr)
    return EXIT_OK


async def cmd_query(args: argparse.Namespace) -> int:
    async with Ragkit() as rag:
        hits = await rag.query(
            args.text,
            top_k=args.top_k,
            mode=args.mode,
            rerank=args.rerank,
            filter_expr=args.filter or "",
        )

    if args.json:
        _emit_json(
            [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "doc_id": h.chunk.doc_id,
                    "score": h.score,
                    "source": h.chunk.metadata.get("source"),
                    "text": h.chunk.text,
                }
                for h in hits
            ]
        )
        return EXIT_OK

    if not hits:
        print("没有检索到相关内容。")
        return EXIT_OK

    print(f"「{args.text}」  {len(hits)} 条结果（mode={args.mode}）\n")
    for rank, hit in enumerate(hits, start=1):
        print(_show(hit, rank))
    return EXIT_OK


async def cmd_count(args: argparse.Namespace) -> int:
    async with Ragkit() as rag:
        total = await rag.count()

    if args.json:
        _emit_json({"count": total})
    else:
        print(f"{total} 条（近似值）")
    return EXIT_OK


async def cmd_delete(args: argparse.Namespace) -> int:
    async with Ragkit() as rag:
        deleted = [await rag.delete_document(doc_id) for doc_id in args.doc_ids]

    if args.json:
        _emit_json({"deleted": dict(zip(args.doc_ids, deleted, strict=True))})
    else:
        for doc_id, count in zip(args.doc_ids, deleted, strict=True):
            print(f"{doc_id}  删除 {count} 条")
    return EXIT_OK


async def cmd_drop(args: argparse.Namespace) -> int:
    settings = get_settings()

    # 删整张表不可逆，默认必须确认；--yes 是给脚本用的
    if not args.yes:
        prompt = f"确定要删掉 collection「{settings.milvus_collection}」吗？这不可恢复。[y/N] "
        if input(prompt).strip().lower() not in ("y", "yes"):
            print("已取消")
            return EXIT_OK

    async with Ragkit() as rag:
        await rag.drop()
    print(f"已删除「{settings.milvus_collection}」")
    return EXIT_OK


async def cmd_eval(args: argparse.Namespace) -> int:
    samples = load_dataset(args.dataset)
    async with Ragkit() as rag:
        report = await evaluate(
            samples,
            rag.retriever,
            k=args.top_k,
            mode=args.mode,
            rerank=args.rerank,
        )

    if args.json:
        _emit_json(report.model_dump())
        return EXIT_OK

    print(report.summary())
    print()
    for name, value in report.metrics.items():
        print(f"  {name:<10} {value:.3f}")
    if report.n_failed:
        # 看指标之前先看失败数：把「接口超时」和「检索质量差」混在一起会误判
        print(f"\n⚠️ {report.n_failed} 条样本失败：")
        for failure in report.failures[:5]:
            print(f"  - {failure.question}: {failure.error}")

    hint = _eval_mismatch_hint(report)
    if hint:
        print(f"\n⚠️ {hint}", file=sys.stderr)
    return EXIT_OK


def cmd_mcp(args: argparse.Namespace) -> int:
    """起 MCP 服务端（stdio）。它会一直阻塞，直到客户端断开或 Ctrl+C。"""
    try:
        from .mcp_server import main as serve
    except ImportError as exc:  # mcp 是可选依赖
        print(
            f"MCP 服务端需要可选依赖：先跑 `uv sync --extra mcp`\n（{exc}）",
            file=sys.stderr,
        )
        return EXIT_ERROR
    serve()
    return EXIT_OK


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于脚本解析")


def _add_retrieval(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-k", "--top-k", type=int, default=None, help="返回几条（默认取配置）")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="dense", help="检索模式")
    parser.add_argument("--rerank", action="store_true", help="开启重排（需要配 RERANK_*）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragkit",
        description="通用 RAG 工具包：解析 / 切分 / 入库 / 检索 / 评估",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="解析文件并报告字数（不入库）")
    p.add_argument("paths", nargs="+", help="文件或目录")
    _add_json(p)
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("split", help="切分单个文件，可选导出 JSONL（不入库）")
    p.add_argument("path", help="文件路径")
    p.add_argument("--out", help="导出成 JSONL，方便肉眼检查")
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--chunk-overlap", type=int, default=None)
    _add_json(p)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("ingest", help="解析 -> 切分 -> 向量化 -> 入库")
    p.add_argument("paths", nargs="+", help="文件或目录")
    p.add_argument("--reset", action="store_true", help="先删掉整张表再建（改了 schema 必须用）")
    p.add_argument("--no-flush", action="store_true", help="跳过写入后的 flush（大批量导入时更快）")
    _add_json(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="检索（终端里手动验证用）")
    p.add_argument("text", help="查询文本")
    p.add_argument("--filter", help='Milvus 过滤表达式，如 doc_id == "abc"')
    _add_retrieval(p)
    _add_json(p)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("count", help="记录数（近似值）")
    _add_json(p)
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("delete", help="按 doc_id 删除文档")
    p.add_argument("doc_ids", nargs="+")
    _add_json(p)
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("drop", help="删掉整张表（不可恢复）")
    p.add_argument("--yes", action="store_true", help="跳过确认（给脚本用）")
    p.set_defaults(func=cmd_drop)

    p = sub.add_parser("eval", help="在评估集上跑一遍，打印指标")
    p.add_argument("dataset", help="JSONL 评估集")
    _add_retrieval(p)
    _add_json(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("mcp", help="起 MCP 服务端（需要 --extra mcp）")
    p.set_defaults(func=cmd_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        outcome = args.func(args)
        # 命令有的是协程、有的是普通函数（比如 mcp 要自己阻塞），
        # 用 iscoroutine 判一下比给每种命令加标记简单。
        if asyncio.iscoroutine(outcome):
            outcome = asyncio.run(outcome)
        return int(outcome)
    except RagkitError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        if exc.source:
            print(f"  来源：{exc.source}", file=sys.stderr)
        return EXIT_ERROR
    except BaseExceptionGroup as exc:
        # 多个文件同时失败时保留分组（那时你确实需要看到全部），但要排版得像人话
        leaves = _leaf_errors(exc)
        print(f"错误：{len(leaves)} 个操作失败", file=sys.stderr)
        for leaf in leaves:
            print(f"  - {leaf}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
