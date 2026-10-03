"""DOCX 解析器，基于 python-docx。

注意：第三方包名就叫 ``docx``，和本模块名字很像。
所以本文件特意命名为 docx_parser.py —— 万一你哪天直接
``python src/ragkit/parsing/docx.py`` 当脚本跑，Python 会把文件所在目录
放进 sys.path，``import docx`` 就会 import 到自己，出现莫名其妙的循环导入。
换个名字就彻底没这个坑。
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

from ..errors import ParseError
from ..schemas import Document
from ..utils import stable_id
from .loaders import run_blocking

__all__ = ["DocxParser"]


def _extract_docx_text(path: Path) -> str:
    """同步抽取 docx 的全部段落文字。

    TODO(你)：
        1) ``doc = DocxDocument(str(path))``
        2) 返回所有段落文字，用两个换行连接：

               return "\\n\\n".join(p.text for p in doc.paragraphs)

        为什么用两个换行：docx 里每个 paragraph 是**语义上的独立段落**。
        用 "\\n\\n" 分隔，M3 按段落切分时才能识别边界；
        只用一个 "\\n" 的话，程序没法区分「换行」和「换段」。

    已知限制（把这段留在 docstring 里，别假装支持）：
        ``doc.paragraphs`` **不包含表格里的文字**。这一版先不处理表格；
        以后要支持，需要另外遍历 ``doc.tables`` 里的 ``cell.text``。
        把限制写清楚，比让用户自己踩坑强。
    """
    doc = DocxDocument(str(path))

    return "\n\n".join(p.text for p in doc.paragraphs)
    # raise NotImplementedError("TODO: 抽取 docx 段落文字")


class DocxParser:
    """处理 .docx 文件。"""

    extensions: tuple[str, ...] = (".docx",)

    async def parse(self, path: Path) -> Document:
        """TODO(你)：和 PdfParser.parse 完全同一个套路。

        1) try/except Exception -> ParseError 的异常翻译，包住
           ``await run_blocking(_extract_docx_text, path)``
        2) 空内容拦截（空文档 / 纯图片文档）
        3) metadata：path / suffix / size_bytes / char_count
           （docx 没有「页数」这个概念，别硬编一个 page_count）
        4) 返回 Document(doc_id=stable_id(str(path), text), ...)

        写完对比一下它和 pdf_parser.py 的重合度 —— 如果两处几乎一模一样，
        那就是 M3 该抽公共函数的信号。**先写重复，再消除重复**，
        别一上来就为了「优雅」抽象出没人看得懂的东西。
        """
        try:
            text = await run_blocking(_extract_docx_text, path)
        except Exception as exc:
            raise ParseError(f"解析 DOCX 失败: {exc}", source=str(path)) from exc

        if not text.strip():
            raise ParseError(
                "DOCX 里没有抽出任何文字",
                source=str(path),
            )

        metadata = {
            "path": str(path),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "char_count": len(text),
        }

        return Document(
            doc_id=stable_id(str(path), text),
            text=text,
            source=str(path),
            metadata=metadata,
        )

        # raise NotImplementedError("TODO: 照 pdf_parser.py 的套路写")
