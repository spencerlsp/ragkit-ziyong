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

    # 针对工程文档：按标题分节切 + 结构感知检索
    uv run ragkit ingest docs/工程文档 --splitter section --reset
    uv run ragkit query "middleware 怎么拦截模型调用" --section-aware --mode hybrid

    # 「我到底在操作哪张表」：先看配置再谈别的
    uv run ragkit info

退出码：``0`` 成功、``1`` 业务失败（RagkitError）、``2`` 参数错误。
**别小看这个区分** —— 很多 CLI 一律 exit(1)，于是脚本里分不清
「参数写错了」和「检索失败了」。

``2`` 要专门说一句：argparse 自己管得住 ``choices`` 和类型（写错命令、
``--mode 乱填``），但它管不住「参数之间」的约束（``overlap >= chunk_size``、
``--section-boost -1``）。那些校验在切分器 / 检索器的构造函数里，
抛出来是 ``SplitError`` / ``RetrievalError`` —— 直接漏到 ``main``
会被当成业务失败（1）。所以 CLI 在边界上把它们重新分类成 ``UsageError``。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .errors import ParseError, RagkitError, RetrievalError
from .evaluation import EvalReport, evaluate, load_dataset
from .parsing import parse_file, parse_files, supported_extensions
from .pipeline import Ragkit
from .processing import clean_document
from .retrieval import Retriever, SectionAwareRetriever
from .schemas import ScoredChunk
from .splitting import MarkdownSectionSplitter, RecursiveSplitter, Splitter, write_chunks_jsonl

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

# ``--section-aware`` 三个旋钮的默认值。
# 放在这里而不是 argparse 的 ``default`` 里，是为了能区分
# 「用户显式传了 4」和「用户没传、默认就是 4」—— 前者说明他知道有这个开关。
SECTION_AWARE_DEFAULTS: dict[str, float] = {
    "candidate_factor": 4,
    "section_boost": 0.5,
    "max_per_section": 2,
}


class UsageError(RagkitError):
    """参数用法错误（退出码 2）。

    为什么定义在 CLI 里而不是 ``ragkit.errors``：这是**命令行的概念** ——
    库的使用者不需要知道「退出码」这回事，把它塞进公共异常体系
    只会让 ``ragkit.errors`` 变成杂物间。它继承 ``RagkitError``，
    所以调用方 ``except ragkit.RagkitError`` 仍然抓得住。
    """


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
        "  第二常见的原因：**入库和标注用的不是同一个切分器** ——"
        " `--splitter section` 和默认的 recursive 切出来的 chunk_id 完全不同。\n"
        "  检查：先跑 `ragkit info` —— 它会把「当前生效的配置 + 库里的条数」"
        "一次打全，包括 splitter / chunk_size / chunk_overlap 和 .env 的实际路径。\n"
        "  确认入库和生成标注用的是同一套这三个值；"
        "或者重新 `ragkit ingest <文档> --reset` 后按当前参数重新标注。"
    )


def _chunk_params(args: argparse.Namespace) -> tuple[int, int]:
    """解析并**校验**切分参数，返回 ``(chunk_size, chunk_overlap)``。

    这里必须用 ``is not None`` 判断，不能用 ``or`` ——
    ``--chunk-overlap 0`` 是合法输入（而且是 section 切分器的默认值），
    用 ``or`` 会把 0 当成"没传"，静默换成配置里的 120。

    两个默认值刻意不同：

      * ``recursive`` 沿用 ``.env`` 的 CHUNK_OVERLAP（默认 120）——
        通用文本按分隔符切，块与块之间留重叠能缓解「一句话被切断」；
      * ``section`` 默认 **0** —— 「一节一块」本身就是自然边界，
        而 ``scripts/build_eval_variants.py`` 生成评估集时的默认值也是 0。
        让两处默认值一致，能直接消掉一整类「标注全对不上」的事故。

    校验放在这里、而不是等切分器构造函数抛，是因为只有 CLI 知道
    这个值是从 ``--chunk-overlap`` 来的还是从 ``.env`` 来的。
    切分器报的那句「chunk_overlap(120) 必须小于 chunk_size(80)」
    不会告诉你 120 是哪来的，用户只能盯着命令行发呆。
    """
    settings = get_settings()
    chunk_size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
    splitter = getattr(args, "splitter", "recursive")

    if args.chunk_overlap is not None:
        chunk_overlap = args.chunk_overlap
        overlap_src = "--chunk-overlap"
    elif splitter == "section":
        chunk_overlap = 0
        overlap_src = "section 切分器的默认值"
    else:
        chunk_overlap = settings.chunk_overlap
        overlap_src = f".env 的 CHUNK_OVERLAP={settings.chunk_overlap}"

    if chunk_size <= 0:
        raise UsageError(f"--chunk-size 必须为正数，收到 {chunk_size}")
    if chunk_overlap < 0:
        raise UsageError(f"--chunk-overlap 不能为负数，收到 {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise UsageError(
            f"chunk_overlap={chunk_overlap}（来自 {overlap_src}）"
            f"必须小于 chunk_size={chunk_size}。"
            f"两个参数一起给就行，例如 --chunk-size {chunk_size} "
            f"--chunk-overlap {max(chunk_size // 5, 1)}"
        )
    return chunk_size, chunk_overlap


def _expand_paths(paths: Sequence[str]) -> list[Path]:
    """把「文件或目录」展开成文件列表。

    帮助里一直写着"文件或目录"，但实现只认文件 —— 于是传目录会报错，
    用户只好在 shell 里自己拼 glob，而 PowerShell 的
    ``(Get-ChildItem ...).FullName`` 和 bash 的 ``*.md`` 语法完全不同，
    换个终端就翻车（这正是本项目里真实发生过的一次）。

    所以直接支持目录：递归子目录、只收解析器认得的后缀、结果排序。
    排序不是洁癖 —— 顺序不稳定的话，同一批文件两次入库的 doc 顺序会变，
    对比结果时就会莫名其妙。
    """
    extensions = set(supported_extensions())
    expanded: list[Path] = []

    for raw in paths:
        path = Path(raw)
        if not path.is_dir():
            expanded.append(path)
            continue

        matched = sorted(
            item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions
        )
        if not matched:
            raise ParseError(
                f"目录里没有可解析的文件（支持的后缀：{', '.join(sorted(extensions))}）",
                source=str(path),
            )
        expanded.extend(matched)

    return expanded


def _build_splitter(args: argparse.Namespace) -> Splitter:
    """按 ``--splitter`` 造切分器。``split`` 和 ``ingest`` 必须走同一套规则。"""
    chunk_size, chunk_overlap = _chunk_params(args)

    if args.splitter != "section":
        # --no-context / --no-merge 只对 section 切分器有意义。
        # 不提醒的话，它们会被静默忽略 —— 而「参数传了却不生效」
        # 正是这个项目里最难自查的一类问题（`--chunk-overlap 0` 那次就是）。
        ignored = [
            flag
            for flag, given in (("--no-context", args.no_context), ("--no-merge", args.no_merge))
            if given
        ]
        if ignored:
            print(
                f"提示：{'、'.join(ignored)} 只在 --splitter section 下生效，"
                "本次已忽略（当前是 recursive）。",
                file=sys.stderr,
            )
        return RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return MarkdownSectionSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_context=not args.no_context,
        merge_small_sections=not args.no_merge,
    )


def _build_retriever(args: argparse.Namespace, rag: Ragkit) -> Retriever | SectionAwareRetriever:
    """按 ``--section-aware`` 决定用通用检索器还是结构感知的后处理层。

    传进去的 ``base`` 是 Ragkit 自己的检索器，所以它**不会被关** ——
    生命周期仍归 Ragkit 管，这里只是借用。
    """
    tuning_given = {
        "--candidate-factor": args.candidate_factor is not None,
        "--section-boost": args.section_boost is not None,
        "--max-per-section": args.max_per_section is not None,
    }

    if not args.section_aware:
        ignored = [flag for flag, given in tuning_given.items() if given]
        if ignored:
            print(
                f"提示：{'、'.join(ignored)} 只在 --section-aware 打开时才生效，本次已忽略。",
                file=sys.stderr,
            )
        return rag.retriever

    def knob(name: str) -> float:
        """取参数值；没传就用文档里写的默认值（见 SECTION_AWARE_DEFAULTS）。"""
        raw = getattr(args, name)
        return SECTION_AWARE_DEFAULTS[name] if raw is None else float(raw)

    try:
        return SectionAwareRetriever(
            base=rag.retriever,
            candidate_factor=int(knob("candidate_factor")),
            section_boost=knob("section_boost"),
            max_per_section=int(knob("max_per_section")),
        )
    except RetrievalError as exc:
        # 参数校验的**唯一真相在 SectionAwareRetriever 里**。CLI 只做一件事：
        # 把分类从「检索失败(1)」改成「参数错误(2)」。在这里再抄一遍校验规则，
        # 等于给自己留一处迟早会漂移的副本。
        raise UsageError(f"--section-aware 的参数不合法：{exc}") from exc


# ---------------------------------------------------------------------------
# 只读、不碰数据库的命令
# ---------------------------------------------------------------------------


async def cmd_parse(args: argparse.Namespace) -> int:
    """解析文件并报告字数，不入库 —— 用来确认「解析这一层」是否正常。"""
    documents = await parse_files(_expand_paths(args.paths))

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
    chunk_size, chunk_overlap = _chunk_params(args)
    splitter = _build_splitter(args)

    document = clean_document(await parse_file(args.path))
    chunks = splitter.split(document)

    if args.out:
        written = await write_chunks_jsonl(chunks, Path(args.out))
        print(f"{written} 块 -> {args.out}")

    if args.json:
        _emit_json(
            {
                "doc_id": document.doc_id,
                "splitter": args.splitter,
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
        f"splitter={args.splitter} chunk_size={chunk_size} overlap={chunk_overlap}）"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# 碰数据库的命令
# ---------------------------------------------------------------------------


async def cmd_ingest(args: argparse.Namespace) -> int:
    splitter = _build_splitter(args)
    async with Ragkit(splitter=splitter) as rag:
        result = await rag.ingest(
            _expand_paths(args.paths), reset=args.reset, flush=not args.no_flush
        )

    if args.json:
        _emit_json(result.model_dump())
        return EXIT_OK

    print(f"入库完成：{result.files} 个文件  {result.chunks} 个 chunk  {result.written} 条记录")
    if result.chunks != result.written:
        # 这三个数本该相等，不等就说明中间某一步吞了东西
        print("⚠️ chunk 数和写入数不一致，检查一下", file=sys.stderr)
    if args.splitter == "section":
        # chunk_id 由切分结果决定：换了切分器，标注也得跟着换
        print(
            "提示：这批数据的 chunk_id 是按 section 切分算出来的；"
            "评估时要用同样按 section 标注的那版评估集，否则指标会全是 0。",
            file=sys.stderr,
        )
    return EXIT_OK


async def cmd_query(args: argparse.Namespace) -> int:
    # 空查询在这里就拦住：让它一路走到 Retriever 里报错的话，
    # 用户看到的是「错误：查询文本不能为空」+ 退出码 1 —— 而这明明是
    # 「参数没给对」，该是 2。分类错了，脚本里的判断就跟着错。
    if not args.text.strip():
        raise UsageError("查询文本不能为空")

    async with Ragkit() as rag:
        retriever = _build_retriever(args, rag)
        hits = await retriever.retrieve(
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

    tail = " + section-aware" if args.section_aware else ""
    print(f"「{args.text}」  {len(hits)} 条结果（mode={args.mode}{tail}）\n")
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


def _info_rows(settings: Settings) -> list[tuple[str, str]]:
    """当前生效的配置，摊成 ``(名字, 值)`` 列表。

    刻意**不打印任何密钥**，只打印「配没配」——
    ``ragkit info`` 的输出是会被贴进 issue 和聊天窗口的。

    ``.env`` 那一行写的是 ``当前工作目录`` 下的 .env：
    pydantic-settings 的 ``env_file=".env"`` 是**相对 CWD** 解析的
    （不是相对 ragkit 包的位置）。所以在别的项目里跑这个 CLI，
    配置读的是那个项目的 .env —— 这行就是用来确认这件事的。
    """
    env_file = Path.cwd() / ".env"
    has_rerank_key = bool(settings.rerank_api_key.get_secret_value())
    rerank_key_state = "已配" if has_rerank_key else "未配（--rerank 会失败）"
    return [
        ("Milvus 地址", settings.milvus_uri),
        ("数据库", settings.milvus_db or "default（MILVUS_DB 为空）"),
        ("Collection", settings.milvus_collection),
        ("向量维度", f"{settings.openai_embedding_dim}（OPENAI_EMBEDDING_DIM）"),
        ("Embedding", f"{settings.openai_embedding_model} @ {settings.openai_embedding_base_url}"),
        ("重排模型", f"{settings.rerank_model}（RERANK_API_KEY {rerank_key_state}）"),
        ("默认切分", f"chunk_size={settings.chunk_size}  chunk_overlap={settings.chunk_overlap}"),
        ("默认检索", f"top_k={settings.top_k}（mode 由命令行的 --mode 决定）"),
        (".env", f"{env_file}（{'存在' if env_file.exists() else '不存在，全部走默认值'}）"),
    ]


# 这几个字段的值不能出现在输出里：`ragkit info` 的结果是会被贴进
# issue、聊天窗口和 CI 日志的。
SECRET_FIELDS = (
    "openai_chat_api_key",
    "openai_embedding_api_key",
    "rerank_api_key",
    "milvus_token",
)


def _settings_dump(settings: Settings) -> dict[str, Any]:
    """**全部**配置项（密钥只留「配没配」），键名与 `.env` 里的写法一一对应。

    为什么「人看的」只挑九行、而「机器看的」给全 24 项：
    人眼要的是「先看这几条就够」，而脚本要的是**能两边 diff**——
    「两个项目里跑的 ragkit 配置一致吗」这个问题，只有全量对比才答得了。
    省略的字段会走代码默认值，所以「谁被省略了」本身也是信息。
    """
    raw: dict[str, Any] = settings.model_dump()
    for field in SECRET_FIELDS:
        secret = getattr(settings, field)
        # SecretStr 的取值要走 get_secret_value()，而且只用来判断「空不空」
        value = secret.get_secret_value() if hasattr(secret, "get_secret_value") else secret
        raw[field] = "已配置" if value else ""
    return dict(sorted(raw.items()))


async def cmd_info(args: argparse.Namespace) -> int:
    """打印当前生效的配置 + 库里的条数 —— 排查「我在哪张表上」的第一步。

    没有这个命令之前，想知道自己连的是哪个库只能靠 ``count`` 猜，
    而「连错了库」和「表是空的」在 count 眼里长得一模一样。

    配置部分是**离线**的（只是把 Settings 打出来），所以即使 Milvus 连不上，
    它也会照常打印 —— 那正是最需要看配置的时候。连不上时退出码是 1，
    脚本里判断这个就够了；想看详情就看 ``--json`` 里的 ``error``。
    """
    settings = get_settings()
    rows = _info_rows(settings)

    total: int | None = None
    error: str | None = None
    try:
        async with Ragkit() as rag:
            total = await rag.count()
    except RagkitError as exc:
        error = str(exc)

    if args.json:
        _emit_json({"settings": _settings_dump(settings), "count": total, "error": error})
        return EXIT_OK if total is not None else EXIT_ERROR

    for key, value in rows:
        print(f"{key}：{value}")
    print()
    if total is None:
        print(f"记录数：取不到（{error}）")
        return EXIT_ERROR
    print(f"记录数：{total} 条（近似值，刚写入还没 flush 的数据可能没算进来）")
    print()
    print("（这里只挑常用的几项；完整的 24 项用 `ragkit info --json` 看，密钥会被掩码）")
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
    # 连库名一起说清楚。两个库里有同名表的时候，光看 collection 名
    # 根本分不出删的是哪一个 —— 而这一步不可恢复。
    where = f"{settings.milvus_uri} / {settings.milvus_db or 'default'}"

    # 删整张表不可逆，默认必须确认；--yes 是给脚本用的
    if not args.yes:
        prompt = (
            f"确定要删掉 {where} 上的 collection"
            f"「{settings.milvus_collection}」吗？这不可恢复。[y/N] "
        )
        if input(prompt).strip().lower() not in ("y", "yes"):
            print("已取消")
            return EXIT_OK

    async with Ragkit() as rag:
        await rag.drop()
    print(f"已删除 {where} 上的「{settings.milvus_collection}」")
    return EXIT_OK


async def cmd_eval(args: argparse.Namespace) -> int:
    samples = load_dataset(args.dataset)
    async with Ragkit() as rag:
        retriever = _build_retriever(args, rag)
        report = await evaluate(
            samples,
            retriever,
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


def _add_chunking(parser: argparse.ArgumentParser) -> None:
    """切分相关的开关。``split`` 与 ``ingest`` 共用同一套 ——
    两边能选的切分器如果不一致，「本地看切分」就失去了意义。"""
    parser.add_argument(
        "--splitter",
        choices=("recursive", "section"),
        default="recursive",
        help="切分器：recursive=通用递归（默认）；section=按 Markdown 标题分节、"
        "每块补 [doc | section] 前缀、代码块不切开、剥掉 frontmatter",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="每块目标字符数（默认取 .env 的 CHUNK_SIZE）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="相邻块重叠字符数（默认：recursive 取 .env 的 CHUNK_OVERLAP，"
        "section 为 0）—— 它决定 chunk_id，入库和评估必须一致",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="只在 --splitter section 下生效：不补 [doc | section] 前缀（消融对照用）",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="只在 --splitter section 下生效：不合并相邻的小节（消融对照用）",
    )


def _add_retrieval(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-k", "--top-k", type=int, default=None, help="返回几条（默认取配置）")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="dense", help="检索模式")
    parser.add_argument("--rerank", action="store_true", help="开启重排（需要配 RERANK_*）")
    parser.add_argument(
        "--section-aware",
        action="store_true",
        help="结构感知后处理：标题加权 + 去近似重复 + 同节限流",
    )
    parser.add_argument(
        "--candidate-factor",
        type=int,
        default=None,
        help="只在 --section-aware 下生效：先取 top_k 的几倍候选再做后处理（默认 4）",
    )
    parser.add_argument(
        "--section-boost",
        type=float,
        default=None,
        help="只在 --section-aware 下生效：标题命中查询词时的最大加成比例（默认 0.5）",
    )
    parser.add_argument(
        "--max-per-section",
        type=int,
        default=None,
        help="只在 --section-aware 下生效：同一节最多返回几条，防止一节刷屏（默认 2）",
    )


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
    _add_chunking(p)
    _add_json(p)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("ingest", help="解析 -> 切分 -> 向量化 -> 入库")
    p.add_argument("paths", nargs="+", help="文件或目录")
    p.add_argument("--reset", action="store_true", help="先删掉整张表再建（改了 schema 必须用）")
    p.add_argument("--no-flush", action="store_true", help="跳过写入后的 flush（大批量导入时更快）")
    _add_chunking(p)
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

    p = sub.add_parser("info", help="当前生效的配置 + 库里的条数（排查用）")
    _add_json(p)
    p.set_defaults(func=cmd_info)

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
    except UsageError as exc:
        # 必须排在 RagkitError 前面：UsageError 是它的子类，
        # 顺序反了就会被上面那条吞掉，退出码永远是 1。
        print(f"用法错误：{exc}", file=sys.stderr)
        return EXIT_USAGE
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
