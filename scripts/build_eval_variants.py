"""把「题目 + 答案依据」的评估集主文件，映射成可以直接跑的评估集。

为什么要有这一步：
    ``chunk_id`` 是**切分结果**的函数（内容哈希），换切分器就全变了。
    如果评估集里写死 chunk_id，那就等于把评估集绑死在一种切分方式上 ——
    想对比「换个切分器效果如何」，就得重新手工标注一遍。

    所以主文件只记 ``question`` + ``evidence``（答案原文片段），
    由这个脚本按当前切分结果把 evidence 映射成 chunk_id：
    **凡是包含答案原文的块都算相关**。切分器换了，重新跑一次脚本即可。

用法::

    # 专属切分器（默认）
    uv run python scripts/build_eval_variants.py \
        --out examples/eng_docs/eval_langchain_section.jsonl

    # 基线切分器，做对照
    uv run python scripts/build_eval_variants.py --splitter recursive \
        --out examples/eng_docs/eval_langchain_recursive.jsonl

拒答题（语料里没有答案）会单独写到 ``<out>.refusals.jsonl``：
它们的 ``relevant_ids`` 是空的，**不能混进检索指标** ——
混进去只会让 hit_rate 全线归零，而不是量出「会不会瞎编」。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ragkit import clean_document, parse_file
from ragkit.config import get_settings
from ragkit.schemas import Chunk
from ragkit.splitting import MarkdownSectionSplitter, RecursiveSplitter

DEFAULT_CORPUS = Path("examples/eng_docs/langchain")
DEFAULT_MASTER = Path("examples/eng_docs/eval_langchain_master.jsonl")


def build_splitter(args: argparse.Namespace) -> Any:
    """按命令行参数造切分器；两种切分器共用的参数走同一套开关。

    **默认值必须和 `ragkit ingest` 完全一致**，否则这个脚本就成了
    「制造对不上标注」的机器：入库用 .env 的 120，生成标注用 0，
    两边的 chunk_id 一个都对不上，指标静默偏低。
    规则抄自 `ragkit.cli._chunk_params`（那边有更详细的注释）：
    recursive 沿用 .env，section 用 0。
    """
    settings = get_settings()
    chunk_size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
    if args.chunk_overlap is not None:
        chunk_overlap = args.chunk_overlap
    elif args.splitter == "section":
        chunk_overlap = 0
    else:
        chunk_overlap = settings.chunk_overlap

    if args.splitter == "recursive":
        return RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return MarkdownSectionSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_context=not args.no_context,
        merge_small_sections=not args.no_merge,
    )


async def split_corpus(corpus: Path, splitter: Any) -> list[Chunk]:
    """把整个语料切一遍，返回全部 chunk。"""
    files = sorted(corpus.glob("*.md"))
    documents = [await parse_file(path) for path in files]
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(splitter.split(clean_document(document)))
    return chunks


def load_master(path: Path) -> list[dict[str, Any]]:
    """读主文件（跳过空行）。"""
    items: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            items.append(json.loads(stripped))
        except json.JSONDecodeError as exc:  # pragma: no cover - 出错就该直接暴露
            raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="生成指定切分配置下的评估集")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="语料目录")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER, help="评估集主文件")
    parser.add_argument("--out", type=Path, required=True, help="输出的评估集 JSONL")
    parser.add_argument(
        "--splitter", choices=["section", "recursive"], default="section", help="用哪种切分器"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="默认取 .env 的 CHUNK_SIZE（和 ragkit ingest 一致）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="默认与 ragkit ingest 一致：recursive 取 .env 的 CHUNK_OVERLAP，section 为 0",
    )
    parser.add_argument("--no-context", action="store_true", help="专属切分器：不补标题前缀")
    parser.add_argument("--no-merge", action="store_true", help="专属切分器：不合并小兄弟节")
    parser.add_argument(
        "--max-hits",
        type=int,
        default=5,
        help="单条依据最多命中多少块；超了就告警（依据太泛 = 标注会虚高）",
    )
    args = parser.parse_args()

    splitter = build_splitter(args)
    chunks = asyncio.run(split_corpus(args.corpus, splitter))
    items = load_master(args.master)

    answerable: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    problems: list[str] = []
    too_broad: list[str] = []
    multi_doc = 0

    print(f"语料 {len(chunks)} 个 chunk；主文件 {len(items)} 条")
    print(f"{'id':<5}{'类型':<10}{'命中块':>6}{'涉及文档':>9}  问题")

    for item in items:
        question = item["question"]
        kind = item.get("type", "fact")
        meta = {"id": item.get("id", ""), "type": kind, "evidence": item.get("evidence", [])}

        if kind == "refusal":
            refusals.append({"question": question, "relevant_ids": [], "metadata": meta})
            print(f"{meta['id']:<5}{kind:<10}{'-':>6}{'-':>9}  {question[:44]}")
            continue

        matched_docs: set[str] = set()
        ids: list[str] = []
        for snippet in item["evidence"]:
            hit = [chunk for chunk in chunks if snippet in chunk.text]
            if not hit:
                problems.append(f"{meta['id']} 依据没命中：{snippet[:60]}")
                continue
            for chunk in hit:
                matched_docs.add(chunk.doc_id)
                if chunk.chunk_id not in ids:
                    ids.append(chunk.chunk_id)

        if not ids:
            problems.append(f"{meta['id']} 完全没有命中任何 chunk，请改依据或删题")
            continue

        if len(matched_docs) > 1:
            multi_doc += 1
        if len(ids) > args.max_hits:
            too_broad.append(
                f"{meta['id']} 命中 {len(ids)} 块（阈值 {args.max_hits}）：{question[:40]}"
            )
        answerable.append({"question": question, "relevant_ids": ids, "metadata": meta})
        print(f"{meta['id']:<5}{kind:<10}{len(ids):>6}{len(matched_docs):>9}  {question[:44]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in answerable),
        encoding="utf-8",
        newline="\n",
    )
    refusals_path = args.out.with_suffix(args.out.suffix + ".refusals.jsonl")
    refusals_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in refusals),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(f"可评题 {len(answerable)} 条 -> {args.out}")
    print(f"拒答题 {len(refusals)} 条 -> {refusals_path}")
    print(f"其中跨文档多跳题 {multi_doc} 条")
    if problems:
        print()
        print("!! 需要修的依据：")
        for line in problems:
            print("   -", line)
        return 1
    if too_broad:
        print()
        print("!! 标注可能过宽（依据太泛，会让指标虚高）：")
        for line in too_broad:
            print("   -", line)
        print("   提示：把依据换成更长的原句，别用短词或标题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
