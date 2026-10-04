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
    """按分隔符优先级递归切分，返回一堆片段（每片长度都可能还超过 chunk_size）。

    TODO(你)：实现它。逻辑如下：

        1) 基线：``if len(text) <= chunk_size: return [text]``
           —— 已经够短了，别再切。

        2) 没有分隔符可用了（``not separators``）：
           说明这段文本里连空格都没有（比如一整串没有标点的中文），
           只能硬切：``[text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]``

        3) 取当前最高级分隔符：``sep, rest = separators[0], separators[1:]``

        4) 如果 ``sep not in text``：这个分隔符在这一段里根本不存在，
           直接降级：``return _split_recursive(text, rest, chunk_size)``

        5) **关键一步**：切开之后把分隔符粘回每段的末尾：

               parts = text.split(sep)
               pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

           为什么要粘回去：这样 ``"".join(pieces) == text`` 恒成立 ——
           切分只允许「切开」，不允许「丢字」。这是切分器的**不变量**，
           有一条测试专门钉它（test_pieces_reassemble_to_original）。
           如果你直接用 text.split(sep)，分隔符就永久丢了，
           最后拼出来的 chunk 会莫名其妙变短，而且极难排查。

           注意 ``parts[:-1]`` 和 ``parts[-1]`` 的区别：最后一段后面没有分隔符了。

        6) 对每个 piece：
               - 长度 <= chunk_size → 直接收下
               - 否则 → ``_split_recursive(piece, rest, chunk_size)`` 递归降级
           收集到一个列表里返回。

    提示：这是**递归**函数，第 4 步和第 6 步都调用了自己。
    递归能停下来的保证是：每次递归时 ``separators`` 都变短了一个，
    最终会走到第 2 步的基线。
    """
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
    """贪心合并：把相邻的小片段拼成尽量接近 chunk_size 的块。

    TODO(你)：一次遍历搞定。

        merged: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > chunk_size:
                merged.append(current)      # 装不下了，先把 current 收走
                current = piece             # 开一个新的，从当前这片开始
            else:
                current += piece            # 还能装下，接着拼
        if current:
            merged.append(current)
        return merged

    两个细节：
      * 拼接用 ``+=``（**不加任何分隔符**），因为每个片段末尾已经粘着自己的
        分隔符了（见 _split_recursive 第 5 步）。再补一个就是重复添加。
      * 循环结束后别忘了把 ``current`` 里剩下的一块收走 ——
        这是「遍历中积累」这类代码最常见的漏写位置。
    """

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
    """给每个 chunk 的开头接上「上一个 chunk 的尾巴」，制造上下文重叠。

    TODO(你)：另一次遍历。

        if overlap <= 0:
            return list(chunks)
        out: list[str] = []
        previous = ""
        for chunk in chunks:
            out.append(previous[-overlap:] + chunk if previous else chunk)
            previous = chunk
        return out

    为什么要 overlap（重叠）：
        假设答案正好横跨两个 chunk 的边界。如果没有重叠，
        两个 chunk 各拿到半句话，检索时谁的相似度都不够高；
        有了重叠，至少有一个 chunk 完整包含那句话。

    代价：加了 overlap 之后，每个 chunk 会比 chunk_size 多出最多 overlap 个字符。
    这是**刻意的**：宁可多带一点上下文，也不要在边界处切断语义。
    所以测试里「chunk 长度 <= chunk_size」这条断言只在 overlap=0 时成立。
    """
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
        """TODO(你)：三件事。

        1) 校验参数，不合法就抛 **SplitError**（不是 ValueError —— M1 建的异常体系就是干这个的）：

               if chunk_size <= 0:
                   raise SplitError(f"chunk_size 必须为正数，收到 {chunk_size}")
               if chunk_overlap < 0:
                   raise SplitError(f"chunk_overlap 不能为负数，收到 {chunk_overlap}")
               if chunk_overlap >= chunk_size:
                   raise SplitError(
                       f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})"
                   )

           最后一条不是形式主义：overlap 大于等于 chunk_size 时，
           FixedSplitter 里的「每次前进多少」会算成 0 或负数，
           程序要么死循环要么原地打转。**在构造时拦住，比在循环里出事便宜得多。**

        2) 存下来：``self.chunk_size = chunk_size``、``self.chunk_overlap = chunk_overlap``

        3) 存下分隔符，**必须拷贝**成新的 list：

               self.separators = (
                   list(separators) if separators is not None else list(DEFAULT_SEPARATORS)
               )

           —— ``list(...)`` 那层拷贝是必须的。如果直接存用户传进来的 list，
           他在外面 append 一个元素，你的切分器行为就跟着悄悄变了。
           **凡是把外部可变对象存成自己状态的地方，都该拷贝一份。**

        参数校验放 __init__ 而不是 split 里：配置错了应该在构造时立刻炸，
        而不是等到处理第 100 个文件时才炸。
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
        """把 Document 切成一批 Chunk。

        TODO(你)：四步组装。

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

        两个要点：

          * ``enumerate`` 的 index 就是这块在文档里的序号。chunk_id 里带上
            doc_id + index + text，保证「同样的输入 → 同样的 id」，
            M5 重复导入时才有稳定的主键可用。

          * Chunk 的校验器会拒绝空白 text。如果 _merge_pieces 或 _apply_overlap
            产出了空字符串，这里会直接抛 ValidationError —— 这正是我们要的：
            **让坏数据在最早的地方暴露。**
        """
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
