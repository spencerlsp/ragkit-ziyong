"""定长切分：最简单也最粗暴的切分器。

留着它主要是当**对照物**：用同一条查询分别在两种切分结果上检索，
你能直观看到「按语义边界切」到底值多少分。
"""

from __future__ import annotations

from ..errors import SplitError
from ..schemas import Chunk, Document
from ..utils import stable_id

__all__ = ["FixedSplitter"]


class FixedSplitter:
    """每 chunk_size 个字符切一刀，相邻块重叠 overlap 个字符。"""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """``chunk_overlap`` 必须小于 ``chunk_size``，否则切分永远不前进。"""
        if chunk_size <= 0:
            raise SplitError(f"chunk_size 必须为正数，收到 {chunk_size}")
        if chunk_overlap < 0:
            raise SplitError(f"chunk_overlap 不能为负数，收到 {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise SplitError(f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # raise NotImplementedError("TODO: 和 RecursiveSplitter 一样的参数校验")

    def split(self, document: Document) -> list[Chunk]:
        """按固定步长滑动窗口切分，相邻块保留 ``chunk_overlap`` 个字符的重叠。

        步长是 ``chunk_size - chunk_overlap``；最后一个窗口可能不足 chunk_size。
        """
        step = self.chunk_size - self.chunk_overlap
        text = document.text
        texts = [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
        return [
            Chunk(
                chunk_id=stable_id(document.doc_id, str(index), text),
                doc_id=document.doc_id,
                text=text,
                index=index,
                metadata={"source": document.source or ""},
            )
            for index, text in enumerate(texts)
        ]
        # raise NotImplementedError("TODO: 滑动窗口切片，见上面的三步")
