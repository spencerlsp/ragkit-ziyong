"""结构感知的 Markdown 切分：专门伺候「工程文档」这类语料。

为什么不能只用 ``RecursiveSplitter``：
    它按 ``\\n\\n`` → ``\\n`` → 句号 一路降级，对**通用文本**很好，
    但工程文档里有三样它看不见的东西：

    1. **标题层级**。``## Streaming`` 底下的内容属于「Streaming 这一节」。
       用分隔符切完，这层归属就没了 —— 命中一段代码，
       却不知道它讲的是哪一节，也没法按节去重。
    2. **代码块**。``` 围起来的代码里随便一个空行都会被当成段落边界，
       切出来的半截代码既读不懂，也不该被当成一条独立知识。
    3. **YAML frontmatter**。文件头那几十行 ``sources: repo://...`` 是元数据，
       不是正文。按普通文本切，它会实实在在占掉 chunk 的位置，
       把向量和 BM25 的注意力吸到「文件路径」上。

这个切分器的做法：

    * 先剥掉 frontmatter，从中取出 ``title`` 当文档标题；
    * 按标题把正文切成「节」，每节记下自己的**标题路径**（breadcrumb）；
    * 每块正文前面补一行 ``[doc: ... | section: A > B]`` ——
      标题里的词因此进入向量与 BM25 的视野，
      这是提升检索命中率最便宜的一招（不花任何模型调用）；
    * 只在节内部按段落继续切，且**代码块永不切开**。

代价要讲清楚：
    * 它比 ``RecursiveSplitter`` 多认一层 Markdown 结构，
      语料不是 Markdown（或标题写得很随意）时收益会缩水；
    * 小节的 chunk 可能明显小于 ``chunk_size`` ——
      这里刻意**不合并**，因为「一节一块」正是检索想要的粒度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import SplitError
from ..schemas import Chunk, Document
from ..utils import stable_id

__all__ = ["MarkdownSectionSplitter"]

# frontmatter：文件开头被两行 --- 夹住的 YAML 块
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.S)

# ATX 标题：# 到 ######，后面必须跟至少一个空格
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")

# 围栏代码块的开/闭：``` 或 ~~~
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")

# frontmatter 里的 title: "xxx"
_TITLE_RE = re.compile(r"""^title:[ \t]*["']?(.+?)["']?[ \t]*$""", re.M)

# 一级标题（用来在没有 frontmatter 时兜底取标题）
_H1_RE = re.compile(r"^#[ \t]+(.+?)[ \t]*$", re.M)

# 断句：中文标点后面直接断；英文句点必须后跟空白才断（避免切开 e.g. / 3.14 / 文件名）
_SENTENCE_RE = re.compile(r"(?<=[。！？；;!?])[ \t]*|(?<=\.)[ \t]+")


@dataclass(slots=True)
class _Section:
    """一个「节」：标题路径 + 正文行（不含标题行本身）。

    ``path`` 里存的是 ``(层级, 标题)`` 序列 —— 层级必须一起留着，
    因为「一级标题被当成文档标题跳过」之后，
    路径长度就不再等于标题层级了（H2 的路径可能只有一个元素）。

    ``path`` 是**展示用**的标题路径，``parent`` 是**合并用**的父路径。
    两者平时一样，只有在「小兄弟节」被并成一块时才分叉：
    展示路径上提到公共父节，父路径保持不变，这样后面还能继续并第三个兄弟。
    """

    path: tuple[tuple[int, str], ...]
    parent: tuple[tuple[int, str], ...]
    lines: list[str]


class MarkdownSectionSplitter:
    """按 Markdown 标题层级切分，并给每块补上标题路径的切分器。"""

    # 短名是 "section" 而不是类名：它要和 CLI 的 --splitter section 对齐，
    # 因为这个名字会被写进入库结果，用于「入库和标注是不是同一套切分」的排查。
    name = "section"

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 0,
        *,
        include_context: bool = True,
        keep_frontmatter: bool = False,
        merge_small_sections: bool = True,
    ) -> None:
        """参数不合法抛 ``SplitError``（与另外两个切分器保持一致的约束）。

        ``include_context``
            是否给每块补 ``[doc: ... | section: ...]`` 这一行。
            关掉它就能干净地对比「补标题」值多少分 —— 专项实验里要的就是这个开关。
        ``keep_frontmatter``
            默认 ``False``（剥掉元数据）。设 ``True`` 则原样保留，
            用来观察那几十行 ``repo://`` 到底干扰了多少检索。
        ``merge_small_sections``
            是否把相邻的「小兄弟节」并成一块。关掉它就能做消融对比：
            「一节一块」和「小节合并」哪种粒度检索更好，用数据说话。
        """
        if chunk_size <= 0:
            raise SplitError(f"chunk_size 必须为正数，收到 {chunk_size}")
        if chunk_overlap < 0:
            raise SplitError(f"chunk_overlap 不能为负数，收到 {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise SplitError(f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_context = include_context
        self.keep_frontmatter = keep_frontmatter
        self.merge_small_sections = merge_small_sections

    # ------------------------------------------------------------------ 对外入口

    def split(self, document: Document) -> list[Chunk]:
        """把 Document 切成一批 Chunk。"""
        front, body = self._split_frontmatter(document.text)
        title = self._document_title(front, body, document.source)

        pieces: list[tuple[tuple[tuple[int, str], ...], str]] = []
        sections = self._iter_sections(body, title)
        if self.merge_small_sections:
            sections = self._coalesce(sections)
        for section in sections:
            for text in self._section_texts(section, title):
                pieces.append((section.path, text))

        return [
            Chunk(
                chunk_id=stable_id(document.doc_id, str(index), text),
                doc_id=document.doc_id,
                text=text,
                index=index,
                metadata={
                    "source": document.source or "",
                    "doc_title": title,
                    "section": " > ".join(name for _, name in path),
                    "heading_level": path[-1][0] if path else 0,
                    "splitter": "markdown_section",
                },
            )
            for index, (path, text) in enumerate(pieces)
        ]

    # ------------------------------------------------------------------ 预处理

    def _split_frontmatter(self, text: str) -> tuple[str, str]:
        """返回 ``(frontmatter, 正文)``。不保留时正文里不会再有元数据。"""
        match = _FRONTMATTER_RE.match(text)
        if match is None or self.keep_frontmatter:
            return "", text
        return match.group(1), text[match.end() :]

    def _document_title(self, front: str, body: str, source: str | None) -> str:
        """标题优先级：frontmatter.title > 正文的一级标题 > 文件名。"""
        match = _TITLE_RE.search(front)
        if match:
            return match.group(1).strip()
        match = _H1_RE.search(body)
        if match:
            return match.group(1).strip()
        if source:
            return Path(source).stem
        return "未命名文档"

    # ------------------------------------------------------------------ 分节

    def _iter_sections(self, body: str, title: str) -> list[_Section]:
        """按标题层级把正文切成节，并记录每节的标题路径。

        围栏代码块内部出现的 ``#`` 一律不当标题 ——
        否则一段 ``# 注释`` 就能把整篇文档的结构带偏。

        **一级标题如果和文档标题相同，就不进路径**：
        它已经写进每块的 ``[doc: ...]`` 前缀了，
        再让 ``Title > Middleware > Hooks`` 出现在每一块里只是重复。
        """
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        in_fence = False

        def flush() -> None:
            path = tuple(stack)
            sections.append(_Section(path=path, parent=path[:-1], lines=list(buffer)))
            buffer.clear()

        for raw in body.splitlines():
            if _FENCE_RE.match(raw):
                in_fence = not in_fence
                buffer.append(raw)
                continue

            match = None if in_fence else _HEADING_RE.match(raw)
            if match is None:
                buffer.append(raw)
                continue

            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1 and heading == title:
                continue
            # 同级或更高级的标题出现时，先把它下面的子标题弹掉
            stack[:] = [item for item in stack if item[0] < level]
            stack.append((level, heading))

        flush()

        # 只有标题、没有正文的空节（常见于目录页）直接丢掉
        return [section for section in sections if any(line.strip() for line in section.lines)]

    # ------------------------------------------------------------------ 合并小兄弟节

    def _coalesce(self, sections: list[_Section]) -> list[_Section]:
        """把相邻的「小兄弟节」并成一块。

        工程文档里到处是只有一两句话的小节（``## Agent State`` 之类）。
        放任它们各成一块的话，会切出一堆 150 字左右的碎片：
        向量信息量太低、还容易互相挤占 ``top_k`` 名额。

        只并**同一个父节下的相邻兄弟**，且加起来要放得进 ``chunk_size`` ——
        跨父节合并会把两段互不相干的内容塞进同一个 chunk，
        那是在制造更难查的召回噪声。
        """
        out: list[_Section] = []
        for section in sections:
            if out:
                previous = out[-1]
                same_parent = section.parent == previous.parent
                # 0.7 是留白：合并后还要加上标题前缀，别正好顶到 chunk_size
                fits = self._size(previous) + self._size(section) <= self.chunk_size * 0.7
                if same_parent and fits:
                    out[-1] = self._merge(previous, section)
                    continue
            out.append(section)
        return out

    @staticmethod
    def _size(section: _Section) -> int:
        """粗略估计一节渲染出来的字符数。"""
        return sum(len(line) + 1 for line in section.lines)

    @staticmethod
    def _heading_line(entry: tuple[int, str]) -> str:
        """按层级还原标题行（合并时要把被剥掉的标题写回正文）。"""
        level, name = entry
        return f"{'#' * level} {name}"

    def _merge(self, first: _Section, second: _Section) -> _Section:
        """把第二个小兄弟节接到第一个后面，展示路径提到公共父节。

        标题行从正文里被剥掉过（改由前缀承载），
        所以合并时必须把各自的标题**写回正文** —— 否则合成的那一块
        只剩正文，读者（和检索）都不知道里面包含哪两个小节。
        已经合并过、路径就等于父路径的块不用再补标题，否则会重复。
        """
        lines: list[str] = []
        for section in (first, second):
            if section.path and section.path != section.parent:
                lines.append(self._heading_line(section.path[-1]))
                lines.append("")
            lines.extend(section.lines)
            lines.append("")
        return _Section(path=second.parent, parent=second.parent, lines=lines)

    # ------------------------------------------------------------------ 成块

    def _section_texts(self, section: _Section, title: str) -> list[str]:
        """把一节变成一块或多块文本，每块都带着 ``[doc | section]`` 前缀。"""
        prefix = self._context_line(title, section.path)
        budget = self.chunk_size - len(prefix) - 2  # 留出前缀和一个空行
        if budget <= 0:
            # 标题路径长到超过 chunk_size 时，前缀不参与预算，避免死循环
            budget = self.chunk_size

        pieces = self._pack(self._blocks(section.lines), budget)
        if self.chunk_overlap > 0 and len(pieces) > 1:
            pieces = self._with_overlap(pieces, self.chunk_overlap)

        if not prefix:
            return pieces
        return [f"{prefix}\n\n{piece}" for piece in pieces]

    def _context_line(self, title: str, path: tuple[tuple[int, str], ...]) -> str:
        """补在每块前面的一行上下文；这是这套切分器的核心收益点。"""
        if not self.include_context:
            return ""
        section = " > ".join(name for _, name in path)
        return f"[doc: {title} | section: {section}]" if section else f"[doc: {title}]"

    @staticmethod
    def _blocks(lines: list[str]) -> list[str]:
        """按空行切块，但**代码块内部的空行不算边界**。"""
        blocks: list[str] = []
        buffer: list[str] = []
        in_fence = False

        for line in lines:
            if _FENCE_RE.match(line):
                in_fence = not in_fence
            if not in_fence and not line.strip():
                if buffer:
                    blocks.append("\n".join(buffer).strip("\n"))
                    buffer = []
                continue
            buffer.append(line)

        if buffer:
            blocks.append("\n".join(buffer).strip("\n"))
        return [block for block in blocks if block.strip()]

    def _pack(self, blocks: list[str], budget: int) -> list[str]:
        """贪心合并相邻块，把每块控制在 ``budget`` 以内。"""
        pieces: list[str] = []
        current = ""

        for block in blocks:
            if len(block) > budget:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(self._split_block(block, budget))
                continue

            candidate = f"{current}\n\n{block}" if current else block
            if len(candidate) <= budget:
                current = candidate
            else:
                pieces.append(current)
                current = block

        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _segments(block: str) -> list[tuple[bool, str]]:
        """把块拆成 ``(是否代码段, 文本)`` 的序列，代码段整体保留。

        为什么需要它：一个块**可能既有说明文字又有代码** ——
        比如「下面这样写：」后面紧跟一段示例。按句子切这种块，
        照样会把代码劈成两半，只是概率比裸切低而已。
        """
        segments: list[tuple[bool, str]] = []
        buffer: list[str] = []
        in_fence = False

        for line in block.splitlines():
            if _FENCE_RE.match(line):
                if in_fence:
                    buffer.append(line)
                    segments.append((True, "\n".join(buffer)))
                    buffer = []
                    in_fence = False
                else:
                    if buffer:
                        segments.append((False, "\n".join(buffer)))
                        buffer = []
                    buffer.append(line)
                    in_fence = True
                continue
            buffer.append(line)

        if buffer:
            segments.append((in_fence, "\n".join(buffer)))
        return segments

    def _split_block(self, block: str, budget: int) -> list[str]:
        """超长块：代码段整体保留，文本段按句子切，再贪心拼回 ``budget`` 以内。"""
        raw: list[str] = []
        for is_code, text in self._segments(block):
            if is_code:
                # 宁可让这一块超长，也不切出半截代码。
                # Milvus 的 text 字段上限是 65535，个别长代码块远到不了。
                raw.append(text)
            else:
                raw.extend(self._split_text(text, budget))

        pieces: list[str] = []
        current = ""
        for piece in raw:
            if not piece.strip():
                continue
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= budget:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = piece
        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _split_text(text: str, budget: int) -> list[str]:
        """纯文本按句子切；单句仍然超长就硬切。"""
        sentences = [piece for piece in _SENTENCE_RE.split(text) if piece.strip()]
        if not sentences:
            return MarkdownSectionSplitter._hard_split(text, budget)

        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > budget:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(MarkdownSectionSplitter._hard_split(sentence, budget))
                continue
            if len(current) + len(sentence) <= budget:
                current += sentence
            else:
                pieces.append(current)
                current = sentence

        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _hard_split(text: str, budget: int) -> list[str]:
        """最后的兜底：按长度硬切，但**尽量在空白处断**。

        直接 ``text[i:i+budget]`` 会把单词劈成两半 ——
        切出来的块以 ``ents and components.`` 开头这种，肉眼一看就是脏数据，
        检索时那个残词还会污染 BM25 的词表。
        """
        pieces: list[str] = []
        remaining = text
        while len(remaining) > budget:
            cut = remaining.rfind(" ", 0, budget)
            if cut <= 0:
                cut = budget  # 整段没有空白（比如一长串代码或 URL），只能硬切
            pieces.append(remaining[:cut])
            remaining = remaining[cut:].lstrip()
        if remaining:
            pieces.append(remaining)
        return pieces

    @staticmethod
    def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
        """只在**同一节内部**的相邻块之间做重叠。

        跨节不重叠：那会把上一节的尾巴接到下一节头上，
        标题路径和正文就对不上了。
        """
        out = [pieces[0]]
        for previous, current in zip(pieces, pieces[1:], strict=False):
            out.append(previous[-overlap:] + "\n" + current)
        return out
