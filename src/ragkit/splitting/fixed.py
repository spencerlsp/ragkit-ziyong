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
        """TODO(你)：校验参数 + 存状态。

        和 RecursiveSplitter.__init__ 的校验完全一样（四条 SplitError），
        但**不要**存 separators —— 这个切分器不需要分隔符。

        写完对比一下两个 __init__ 的重合度。等 M3 收尾时我会让你决定：
        是抽一个公共的校验函数，还是就这么重复着。**现在先重复。**
        """
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
        """TODO(你)：滑动窗口。

        1) ``step = self.chunk_size - self.chunk_overlap``
           —— 这是「每一刀前进多少」。overlap 越大，step 越小，切出来的块越多。

        2) 起点用 ``range(0, len(text), step)``，
           每块切片 ``text[i : i + self.chunk_size]``

        3) 包装成 Chunk（chunk_id / doc_id / index / metadata，和
           RecursiveSplitter.split 里完全一样）

        想清楚一件事：``step`` 有可能是 0 吗？
        如果可能，你的 ``range(0, len(text), 0)`` 会抛什么错？
        —— 想通这个，你就明白 __init__ 里那条 ``chunk_overlap < chunk_size``
        的校验到底在防什么了。

        最后一个窗口可能不足 chunk_size，这是正常的，不用补齐。
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
