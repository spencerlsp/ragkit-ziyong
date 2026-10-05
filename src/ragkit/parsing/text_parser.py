"""纯文本 / Markdown 解析器。

最简单的解析器，先把「解析器长什么样」跑通。
后面的 PDF / DOCX 只是把读取方式换成 run_blocking 而已。
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import Document
from ..utils import stable_id
from .loaders import read_text_async

__all__ = ["TextParser"]


class TextParser:
    """处理 .txt / .md 这类本来就是纯文本的文件。"""

    # 必须显式标注 tuple[str, ...]，不能靠推断。
    # 原因见 base.py 的 Parser 协议：协议里的可变属性是不变（invariant）的，
    # 不写注解时 mypy 会推断成 tuple[str, str, str]，虽然它是子类型，但不满足不变性。
    extensions: tuple[str, ...] = (".txt", ".md", ".markdown")

    async def parse(self, path: Path) -> Document:
        """读取文本并包装成 Document。"""
        text = await read_text_async(path)

        metadata = {
            "path": str(path),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        }
        return Document(
            doc_id=stable_id(str(path), text),
            text=text,
            source=str(path),
            metadata=metadata,
        )

        # raise NotImplementedError("TODO: 参考上面的三步走")
