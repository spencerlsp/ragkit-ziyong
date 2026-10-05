"""切分：把 Document 切成一批 Chunk。

⚠️ 和 parsing 包不同，这个包的核心是**同步**的。
   切分是纯字符串运算，不会等待任何东西 —— 详细理由见 processing.py 顶部。

   唯一的例外是 export.py 里的写文件函数，那是真 IO，所以是 async。
   同一个项目里两种风格并存不是混乱，是**按需选择**。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..config import get_settings
from ..schemas import Chunk, Document
from .export import write_chunks_jsonl
from .fixed import FixedSplitter
from .recursive import RecursiveSplitter

__all__ = [
    "Splitter",
    "RecursiveSplitter",
    "FixedSplitter",
    "split_document",
    "split_documents",
    "write_chunks_jsonl",
]


@runtime_checkable
class Splitter(Protocol):
    """切分器协议。

    注意和 parsing 包的 Parser 对比

        Parser:   async def parse(...) -> Document     ← 要读磁盘，会等待
        Splitter:      def split(...) -> list[Chunk]   ← 只算字符串，不等待

    唯一的差别就是那个 ``async``，而它背后是一条判断标准：
    **这个操作会不会让出控制权（等待 IO / 网络 / 锁）？**
    会 → async；不会 → sync。

    给纯 CPU 函数套 async 是常见的误用：调用方要写 await、
    要到处传 async 上下文，换来的并发收益是**零**（因为它根本不等待）。

    协议只有方法、没有数据属性，所以这次的实现类不会遇到
    M2 那个「tuple 不变性」的坑 —— 方法是函数类型，本来就支持子类型。
    """

    def split(self, document: Document) -> list[Chunk]:
        """把 document 切成一批 Chunk。"""
        ...


def _default_splitter() -> Splitter:
    """按配置构造默认切分器。"""
    settings = get_settings()
    return RecursiveSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def split_document(document: Document, *, splitter: Splitter | None = None) -> list[Chunk]:
    """切分单个 Document。"""
    if splitter is None:
        splitter = _default_splitter()
    return splitter.split(document)
    # raise NotImplementedError("TODO: 两行，见上面的代码")


def split_documents(
    documents: Iterable[Document],
    *,
    splitter: Splitter | None = None,
) -> list[list[Chunk]]:
    """切分一批 Document，返回「每个文档一个 Chunk 列表」。"""
    if splitter is None:
        splitter = _default_splitter()
    return [splitter.split(document) for document in documents]
    # raise NotImplementedError("TODO: 解析一次 splitter，再列表推导")
