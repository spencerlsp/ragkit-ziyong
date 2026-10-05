"""端到端评估 demo：切分 -> 入库 -> 多种检索策略对比。

用法::

    uv run python scripts/eval_demo.py examples/sample_doc.md
    uv run python scripts/eval_demo.py examples/sample_doc.md --rerank
    uv run python scripts/eval_demo.py examples/sample_doc.md --chunk-size 600 --top-k 5

它做四件事：
    1. 解析 + 清洗 + 按指定粒度切分；
    2. **自动标注**：每个问题配一个答案关键词，关键词出现在哪些 chunk 里，
       那些 chunk 就算「相关内容」；
    3. 入库（先按 doc_id 删除再写，所以重跑是幂等的）；
    4. 用同一套样本分别跑稠密 / 混合 /（可选）混合+重排，打印指标对比。

⚠️ 关于自动标注的偏差，必须说清楚：
    它衡量的其实是「有没有捞到**包含那个关键词**的块」。
    这对关键词匹配（BM25）天然友好，对语义匹配（稠密向量）其实更严格 ——
    因为稠密只做对了语义、没用上那个词，也会被判成没命中。
    所以**不要用这个结论去下「混合一定更好」的判断**。
    严谨的做法是人工标注「这一块是否真的回答了问题」，然后冻结成 JSONL。
    自动标注的价值在于：**先把评估流程跑通**，让你看见指标长什么样。

另一个常见误区：换切分参数（chunk_size）会让 chunk_id 全变，
你之前冻结的标注就全部失效了。这也是为什么真实项目里
**评估集要和切分策略一起版本化**。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ragkit import (
    MilvusIndexer,
    Retriever,
    clean_document,
    create_embedder,
    create_reranker,
    evaluate,
    get_settings,
    ingest_chunks,
    parse_file,
)
from ragkit.evaluation import EvalReport, EvalSample, SampleResult
from ragkit.schemas import Chunk
from ragkit.splitting import RecursiveSplitter

# 问题 + 答案里那个「独一无二」的关键词 + 分组。
#
# ★ 分组是这份评估集最重要的设计 ★
# 把所有问题混在一起算平均，你只会得到一个「综合分」，
# 而综合分永远没法告诉你「该不该保留混合检索 / 该不该加重排」。
# 拆成直问 / 改述 / 稀有词三组之后，规律才浮得出来。
#
# 关键词要短、要独特 —— 太长的话可能正好跨在切分边界上，标注会失效。
QUESTION_SPEC: list[tuple[str, str, str]] = [
    # ---- 直问：问题和文档用同一批词，各种方法都该轻松命中 ----
    ("远程办公每周最多几天？", "两天", "直问"),
    ("年假满五年有多少天？", "15 个工作日", "直问"),
    ("报销每月几号截止？", "25 日", "直问"),
    ("笔记本电脑的折旧期是多久？", "折旧期三年", "直问"),
    # ---- 改述：换成同义词，和文档几乎没有共同词汇 ----
    # 这一组是稠密向量的主场：它靠语义而不是字面匹配。
    ("公司允许员工在家上班的上限是几天？", "两天", "改述"),
    ("干满五年之后，一年能休多少天带薪假？", "15 个工作日", "改述"),
    ("每个月什么时候必须把发票交上去？", "25 日", "改述"),
    ("离开公司后，多长时间内不能去竞争对手那里？", "12 个月", "改述"),
    # ---- 稀有词：问题里带文档中的缩写或精确数字 ----
    # 这一组是 BM25 的主场：罕见 token 的精确匹配，向量模型往往抓不住。
    ("PIP 是什么意思？", "绩效改进计划", "稀有词"),
    ("一线城市住宿费每晚的上限是多少？", "600 元", "稀有词"),
    ("出差坐高铁能坐几等座？", "高铁二等座", "稀有词"),
]

GROUPS = ("直问", "改述", "稀有词")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ragkit 评估 demo")
    parser.add_argument("path", type=Path, help="作为知识库的文件")
    parser.add_argument("--chunk-size", type=int, default=350, help="切分粒度（默认 350）")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="切分重叠（默认 50）")
    parser.add_argument("--top-k", type=int, default=3, help="检索前 k 条（默认 3）")
    parser.add_argument("--rerank", action="store_true", help="额外跑一遍「混合 + 重排」")
    return parser.parse_args()


def build_samples(chunks: list[Chunk], spec: list[tuple[str, str, str]]) -> list[EvalSample]:
    """按关键词自动标注。标不到的直接报错，不产废样本。"""
    samples: list[EvalSample] = []
    missing: list[tuple[str, str]] = []
    for question, keyword, group in spec:
        ids = [chunk.chunk_id for chunk in chunks if keyword in chunk.text]
        if not ids:
            missing.append((question, keyword))
            continue
        # 分组存在 metadata 里 —— 评估样本的 metadata 就是派这个用场的
        samples.append(EvalSample(question=question, relevant_ids=ids, metadata={"group": group}))

    if missing:
        # 「标注体检」：不合格的标注必须当场暴露。
        # 默默跳过的话，你会得到一份「指标很好」的报告 ——
        # 因为那些没有标准答案的题目根本没参与评估，分母被悄悄改小了。
        lines = "\n".join(f"  - {q}（关键词「{k}」）" for q, k in missing)
        raise SystemExit(
            f"以下问题的关键词在所有 chunk 里都找不到，标注无效：\n{lines}\n"
            "可能是文档改了，也可能是关键词正好跨在切分边界上。修好再跑。"
        )
    return samples


def group_stats(
    report: EvalReport, spec: list[tuple[str, str, str]]
) -> dict[str, dict[str, float]]:
    """按问题分组统计指标。综合分看不出规律，分组能。"""
    group_of = {question: group for question, _, group in spec}
    buckets: dict[str, list[SampleResult]] = {}
    for row in report.results:
        buckets.setdefault(group_of[row.question], []).append(row)

    stats: dict[str, dict[str, float]] = {}
    for name, rows in buckets.items():
        count = len(rows)
        stats[name] = {
            "n": float(count),
            "hit_rate": sum(r.hit_rate for r in rows) / count,
            "mrr": sum(r.rr for r in rows) / count,
            "ndcg": sum(r.ndcg for r in rows) / count,
        }
    return stats


def print_overall(reports: list[tuple[str, EvalReport]]) -> None:
    labels = [label for label, _ in reports]
    print(f"{'指标':<12}" + "".join(f"{label:>13}" for label in labels))
    print("-" * 60)
    for name in ("hit_rate", "recall", "mrr", "ndcg"):
        row = f"{name:<12}"
        for _, report in reports:
            row += f"{report.metrics[name]:>13.3f}"
        print(row)


def print_group_ndcg(reports: list[tuple[str, EvalReport]]) -> None:
    """分组只看 NDCG —— 它同时反映「捞到几个」和「排得怎么样」。"""
    labels = [label for label, _ in reports]
    sets = [group_stats(report, QUESTION_SPEC) for _, report in reports]
    print("\n分组 NDCG（组内平均）")
    print(f"{'分组':<10}{'题数':>6}" + "".join(f"{label:>13}" for label in labels))
    print("-" * 60)
    for group in GROUPS:
        first = sets[0].get(group)
        if first is None:
            continue
        row = f"{group:<10}{int(first['n']):>6}"
        for stats in sets:
            row += f"{stats[group]['ndcg']:>13.3f}"
        print(row)


def print_differences(reports: list[tuple[str, EvalReport]], samples: list[EvalSample]) -> None:
    base_label, base = reports[0]
    truth_of = {sample.question: sample.relevant_ids for sample in samples}

    for label, report in reports[1:]:
        print(f"\n逐题对比：{base_label} vs {label}（只列排名不同的）")
        print("-" * 66)
        shown = 0
        for before, after in zip(base.results, report.results, strict=True):
            if before.rr == after.rr:
                continue
            shown += 1
            print(f"Q: {before.question}")
            print(f"   {base_label} rr={before.rr:.3f}   {label} rr={after.rr:.3f}")
            print(f"   正确答案：{truth_of[before.question]}")
        if not shown:
            print("（没有差异）")


def print_common_misses(reports: list[tuple[str, EvalReport]], samples: list[EvalSample]) -> None:
    """所有模式都没命中的题 —— 最值得看的一类。

    ⚠️ 必须是**所有模式**的交集。只按某一个模式筛的话，
    会把「那个模式失败、但另一个模式救回来了」的题也列进来，
    而那类恰恰是另一种方法的价值所在，混在这里看会得出完全相反的结论。
    """
    truth_of = {sample.question: sample.relevant_ids for sample in samples}
    miss_sets = [{row.question for row in report.results if row.rr == 0.0} for _, report in reports]
    missed_by_all = set.intersection(*miss_sets) if miss_sets else set()

    print("\n所有模式都没命中的题")
    print("-" * 66)
    if not missed_by_all:
        print("（没有）")
    for row in reports[0][1].results:
        if row.question in missed_by_all:
            print(f"Q: {row.question}")
            print(f"   正确答案：{truth_of[row.question]}")


async def main() -> None:
    args = parse_args()
    settings = get_settings()

    embedder = create_embedder(settings)
    indexer = MilvusIndexer(settings)
    reranker = create_reranker(settings) if args.rerank else None
    try:
        # ---- 1~3. 解析 -> 清洗 -> 切分 -> 自动标注 -> 入库 ----
        document = await parse_file(args.path)
        cleaned = clean_document(document)
        splitter = RecursiveSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        chunks = splitter.split(cleaned)
        samples = build_samples(chunks, QUESTION_SPEC)

        print("=" * 66)
        print(f"文档        {args.path.name}  {cleaned.char_count} 字")
        print(f"切分        {args.chunk_size}/{args.chunk_overlap} -> {len(chunks)} 块")
        print(f"评估集      {len(samples)} 条（关键词自动标注）")
        print(f"collection  {settings.milvus_collection}  @ {settings.milvus_uri}")
        print("=" * 66)

        written = await ingest_chunks(chunks, embedder=embedder, indexer=indexer, flush=True)
        print(f"入库        {written} 条  （collection 现有 {await indexer.count()} 条）\n")

        # ---- 4. 逐个策略各跑一遍 ----
        # 【为什么把报告放进列表而不是写成 dense / hybrid 两个变量】
        # 加一个「重排」列意味着要改所有表格代码。写成列表驱动之后，
        # 以后再加一种检索策略只需要往这个列表里加一行。
        modes: list[tuple[str, str, bool]] = [
            ("稠密", "dense", False),
            ("混合", "hybrid", False),
        ]
        if args.rerank:
            modes.append(("混合+重排", "hybrid", True))

        retriever = Retriever(
            embedder=embedder, indexer=indexer, reranker=reranker, settings=settings
        )
        reports: list[tuple[str, EvalReport]] = []
        for label, mode, use_rerank in modes:
            report = await evaluate(samples, retriever, k=args.top_k, mode=mode, rerank=use_rerank)
            reports.append((label, report))

        print(f"top_k = {args.top_k}，共 {len(chunks)} 块\n")
        print_overall(reports)
        print_group_ndcg(reports)

        for label, report in reports:
            if report.n_failed:
                print(f"\n⚠️ {label} 有 {report.n_failed} 条样本失败：")
                for failure in report.failures[:3]:
                    print(f"   - {failure.question}: {failure.error}")

        print_differences(reports, samples)
        print_common_misses(reports, samples)
    finally:
        await indexer.aclose()
        await embedder.aclose()
        if reranker is not None:
            await reranker.aclose()


if __name__ == "__main__":
    asyncio.run(main())
