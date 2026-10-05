"""文件解析层：把磁盘上的文件变成 Document。

对外只有两个入口：
    await parse_file(path)            解析单个文件 -> Document
    await parse_files([p1, p2, ...])  并发解析多个文件 -> list[Document]（顺序与输入一致）

再加一个 register(parser) 用来挂自定义解析器。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from ..config import get_settings
from ..errors import ParseError
from ..schemas import Document
from .base import Parser, get_parser, register, supported_extensions
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .text_parser import TextParser

__all__ = ["Parser", "parse_file", "parse_files", "register", "supported_extensions"]

# 注册内置解析器。
# 放在这里（而不是各个解析器模块内部自动注册）是为了「到底装了哪些解析器」一眼可见。
register(TextParser())
register(PdfParser())
register(DocxParser())


async def parse_file(path: str | Path) -> Document:
    """解析单个文件。"""
    p = Path(path)

    if not p.is_file():
        raise ParseError(f"文件不存在或不是普通文件: {p}", source=str(path))

    parser = get_parser(p)

    return await parser.parse(p)

    # raise NotImplementedError("TODO: 校验文件 -> 查解析器 -> await 调用")


async def parse_files(
    paths: Iterable[str | Path],
    *,
    max_concurrency: int | None = None,
) -> list[Document]:
    """并发解析多个文件。"""
    items = [Path(p) for p in paths]

    results: list[Document | None] = [None] * len(items)

    limit = max_concurrency or get_settings().max_concurrency
    sem = asyncio.Semaphore(limit)

    async def one(index: int, path: Path) -> None:
        async with sem:  # ← 并发闸门，必须在协程内部
            results[index] = await parse_file(path)

    async with asyncio.TaskGroup() as tg:
        for i, p in enumerate(items):
            tg.create_task(one(i, p))

    return [doc for doc in results if doc is not None]
    # raise NotImplementedError("TODO: TaskGroup + Semaphore，保持输出顺序")
