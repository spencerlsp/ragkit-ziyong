"""端到端 demo：解析 -> 清洗 -> 切分 -> 向量化 -> 写入 Milvus -> 验证检索。

用法::

    # 完整流程（会调用 SiliconFlow 的 embedding 接口，需要联网 + API Key）
    uv run python scripts/ingest_demo.py examples/sample_doc.md

    # schema 变了（比如后来给 collection 加了 BM25 稀疏字段）时先重置
    uv run python scripts/ingest_demo.py examples/sample_doc.md --reset

    # 不联网：用本地假向量，只验证「数据库那一段」通不通
    uv run python scripts/ingest_demo.py examples/sample_doc.md --fake

    # 换个验证问题
    uv run python scripts/ingest_demo.py examples/sample_doc.md --query "报销多久能到账"

为什么这个脚本值得留着：
    它是整条流水线**唯一**的端到端验证。165 条单元测试全是离线的，
    它们证明逻辑正确，但证明不了「你的 Milvus 版本支持 BM25」
    「schema 能被服务端接受」「Milvus 返回的命中结构和我们假设的一致」。
    每次改动 schema 或检索逻辑，跑一遍它，心里才有底。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pymilvus import MilvusClient

from ragkit import (
    MilvusIndexer,
    Retriever,
    clean_document,
    create_embedder,
    get_settings,
    ingest_chunks,
    parse_file,
    split_document,
)
from ragkit.embedding import HashingEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ragkit 端到端 demo")
    parser.add_argument("path", type=Path, help="要入库的文件（.md / .txt / .pdf / .docx）")
    parser.add_argument("--reset", action="store_true", help="先删掉整个 collection 再重建")
    parser.add_argument("--fake", action="store_true", help="用本地假向量，不调 embedding 接口")
    parser.add_argument("--query", default="远程办公每周最多几天？", help="用来验证检索的问题")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()

    print("=" * 64)
    print(f"Milvus      {settings.milvus_uri}")
    print(f"collection  {settings.milvus_collection}")
    model = (
        "HashingEmbedder（本地假向量，无语义）" if args.fake else settings.openai_embedding_model
    )
    print(f"embedding   {model}")
    print(f"维度        {settings.openai_embedding_dim}")

    # 先探一下服务端版本 —— BM25 / hybrid_search 要 2.5 以上，
    # 装错版本的话，后面的报错会非常难懂，不如一开始就说清楚。
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    try:
        print(f"服务端      Milvus {client.get_server_version()}")
    finally:
        client.close()
    print("=" * 64)

    embedder = (
        HashingEmbedder(dim=settings.openai_embedding_dim)
        if args.fake
        else create_embedder(settings)
    )
    indexer = MilvusIndexer(settings)
    try:
        if args.reset:
            await indexer.drop()
            print("\n[重置] 已删除旧的 collection")

        # ---- 1. 解析：文件 -> Document ----
        document = await parse_file(args.path)
        print(f"\n[解析] {args.path.name}  {document.char_count} 字")

        # ---- 2. 清洗：归一化空白 ----
        cleaned = clean_document(document)
        print(
            f"[清洗] {cleaned.char_count} 字"
            f"（原始 {cleaned.metadata.get('original_char_count', '?')} 字）"
        )

        # ---- 3. 切分：Document -> list[Chunk] ----
        chunks = split_document(cleaned)
        if not chunks:
            print("[切分] 没切出任何 chunk，看看文件是不是空的")
            return
        sizes = [chunk.char_count for chunk in chunks]
        print(
            f"[切分] {len(chunks)} 块  "
            f"长度 min={min(sizes)} max={max(sizes)} 平均={sum(sizes) // len(sizes)}"
        )

        # ---- 4. 向量化 + 写入（内部是「先按 doc_id 删除，再 upsert」）----
        written = await ingest_chunks(chunks, embedder=embedder, indexer=indexer)
        print(f"[写入] {written} 条")

        total = await indexer.count()
        print(f"[统计] collection 里现在有 {total} 条（近似值，刚写的可能还没落盘）")

        # ---- 5. 验证检索：稠密 vs 混合 ----
        # 这里两个组件都是注入的，所以 Retriever 不会关它们；
        # 由外面的 finally 统一收尾。
        retriever = Retriever(embedder=embedder, indexer=indexer, settings=settings)
        print(f"\n检索问题：{args.query}")

        for label, mode in (("稠密", "dense"), ("混合", "hybrid")):
            try:
                hits = await retriever.retrieve(args.query, top_k=3, mode=mode)
            except Exception as exc:  # noqa: BLE001 - demo 脚本，失败也要继续看另一路
                print(f"\n[{label}] 失败：{type(exc).__name__}: {exc}")
                continue
            print(f"\n[{label}] 前 {len(hits)} 条：")
            for rank, hit in enumerate(hits, start=1):
                # 用 split() + join 压平空白：完全避开了换行符的转义问题
                preview = " ".join(hit.chunk.text.split())[:56]
                print(f"  {rank}. score={hit.score:.4f}  {preview}…")
    finally:
        await indexer.aclose()
        await embedder.aclose()


if __name__ == "__main__":
    asyncio.run(main())
