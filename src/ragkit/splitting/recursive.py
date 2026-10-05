"""递归字符切分：本项目的主力切分器。

思路（LangChain 的 RecursiveCharacterTextSplitter 就是这么干的）：
    按「语义层级」从高到低依次尝试分隔符：段落 > 换行 > 句号 > 逗号 > 字符。
    某一段切完还是太长，就降到下一级分隔符继续切。
    这样切出来的 chunk 尽量落在自然边界上，而不是从句子中间劈开。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import SplitError
from ..schemas import Chunk, Document
from ..utils import stable_id

__all__ = ["RecursiveSplitter", "DEFAULT_SEPARATORS"]

# 分隔符按「语义层级」从高到低排列。
# 注意最后的 " "：中文里没有空格，所以中文长句会一路降到"空段落"再硬切。
# 这是已知的局限 —— 中文按句子切需要标点级的分句（M3 进阶再做）。
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]


def _split_recursive(text: str, separators: Sequence[str], chunk_size: int) -> list[str]:
    """按分隔符优先级递归切分，返回一堆片段（每片长度都可能还超过 chunk_size）。"""
    # 基线条件：文本足够短直接返回
    if len(text) <= chunk_size:
        return [text]

    # 没有可用分隔符，强制硬切
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]

    # 当前分隔符不存在，降级到下一级分隔符
    if sep not in text:
        return _split_recursive(text, rest, chunk_size)

    # 切分并把分隔符补回去，保证拼接还原原始文本
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

    splited: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            splited.append(piece)
        else:
            # 递归切分长片段，并把结果合并进列表
            sub_chunks = _split_recursive(piece, rest, chunk_size)
            splited.extend(sub_chunks)

    return splited
    # raise NotImplementedError("TODO: 见上面的六步")


def _merge_pieces(pieces: Sequence[str], chunk_size: int) -> list[str]:
    """贪心合并：把相邻的小片段拼成尽量接近 chunk_size 的块。"""

    merged: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            merged.append(current)
            current = piece
        else:
            current += piece
    if current:
        merged.append(current)
    return merged
    # raise NotImplementedError("TODO: 贪心合并，见上面的代码骨架")


def _apply_overlap(chunks: Sequence[str], overlap: int) -> list[str]:
    """给每个 chunk 的开头接上「上一个 chunk 的尾巴」，制造上下文重叠。"""
    if overlap <= 0:
        return list(chunks)
    out: list[str] = []
    previous = ""
    for chunk in chunks:
        out.append(previous[-overlap:] + chunk if previous else chunk)
        previous = chunk
    return out
    # raise NotImplementedError("TODO: 见上面的代码骨架")


class RecursiveSplitter:
    """按分隔符优先级递归切分的切分器（本项目默认用它）。"""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Sequence[str] | None = None,
    ) -> None:
        """参数不合法抛 ``SplitError``；``chunk_overlap`` 必须小于 ``chunk_size``。

        ``separators`` 会被拷贝一份 —— 直接存外部传进来的列表的话，
        对方在外面改一下，切分器的行为就跟着变了。
        """
        if chunk_size <= 0:
            raise SplitError(f"chunk_size 必须为正数，收到 {chunk_size}")
        if chunk_overlap < 0:
            raise SplitError(f"chunk_overlap 不能为负数，收到 {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise SplitError(f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators is not None else list(DEFAULT_SEPARATORS)

        # raise NotImplementedError("TODO: 校验参数 + 存状态 + 拷贝分隔符")

    def split(self, document: Document) -> list[Chunk]:
        """把 Document 切成一批 Chunk。"""
        pieces = _split_recursive(document.text, self.separators, self.chunk_size)
        merged = _merge_pieces(pieces, self.chunk_size)
        texts = _apply_overlap(merged, self.chunk_overlap)
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
        # raise NotImplementedError("TODO: 四步组装，见上面的代码")
