"""PDF 解析器，基于 pypdf。

本文件是「同步内核 + 异步外壳」套路的第一个完整示例：
    _extract_pdf_text() 是纯同步函数（第三方库不认 async）
    PdfParser.parse()   是异步外壳，用 run_blocking 把内核丢进线程池
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ..errors import ParseError
from ..schemas import Document
from ..utils import stable_id
from .loaders import run_blocking

__all__ = ["PdfParser"]


def _extract_pdf_text(path: Path) -> tuple[str, int]:
    """同步地把 PDF 每页文字抽出来，返回 (全文, 页数)。

    TODO(你)：实现这个**同步**函数（注意：它故意不是 async）。

    步骤：
        1) ``reader = PdfReader(str(path))``
           —— pypdf 收路径字符串，不是 Path 对象。

        2) 逐页取文字并拼接：

               pages = []
               for page in reader.pages:
                   pages.append(page.extract_text() or "")
               text = "\\n\\n".join(pages)

           ``or ""`` 兜底是必须的：整页是图片（扫描件）时
           ``extract_text()`` 返回 None，不兜底的话后面拼接直接 TypeError。
           用 "\\n\\n" 连接是为了保留页边界，M3 切分时用得上。

        3) 返回 ``(text, len(reader.pages))``

    为什么要拆成独立函数：
      pypdf 是纯同步的阻塞库。统一套路是「同步内核 + 异步外壳」：
      内核干真正的活（这里），外壳只负责把它丢进线程池（下面的 parse）。
      这样内核 100% 可测、可复用，也不掺任何 asyncio 概念。
    """
    reader = PdfReader(str(path))

    # 逐页提取文字并拼接
    pages = []
    for page in reader.pages:
        # ``or ""`` 兜底是必须的：整页是图片（扫描件）时 extract_text() 返回 None，
        # 不兜底的话后面拼接直接 TypeError。
        pages.append(page.extract_text() or "")
    # 用两个换行连接，保留页边界，M3 切分时用得上。
    text = "\n\n".join(pages)
    return (text, len(reader.pages))

    # raise NotImplementedError("TODO: 用 pypdf 逐页抽文字，返回 (全文, 页数)")


class PdfParser:
    """处理 .pdf 文件。"""

    extensions: tuple[str, ...] = (".pdf",)

    async def parse(self, path: Path) -> Document:
        """TODO(你)：四步。

        1) 调用内核并做异常翻译：

               try:
                   text, page_count = await run_blocking(_extract_pdf_text, path)
               except Exception as exc:
                   raise ParseError(f"解析 PDF 失败: {exc}", source=str(path)) from exc

           为什么敢用宽泛的 ``except Exception``：这是**库的边界**。
           pypdf 会抛 PdfReadError / PdfStreamError 等一堆自有异常，
           我们不能指望用户 import pypdf 来 except。所以在边界一次性
           「翻译」成 ragkit 的 ParseError。
           （KeyboardInterrupt / SystemExit 继承自 BaseException，不会被捕获。）

        2) 拦扫描件：

               if not text.strip():
                   raise ParseError(
                       "PDF 里没有抽出任何文字，可能是扫描件（需要 OCR 才能用）",
                       source=str(path),
                   )

           这一步很值钱。扫描版 PDF 抽出来就是空字符串，如果不在这里拦住，
           它会一路走到切分层切出一堆空 chunk，最后在 Milvus 里才暴露 ——
           那时你面对的是几万条空记录，排查成本高得多。
           **在入口处拒绝坏数据，比在下游到处防它便宜得多。**

        3) 组装 metadata，比 TextParser 多两个字段：

               {
                   "path": str(path),
                   "suffix": ".pdf",
                   "size_bytes": path.stat().st_size,
                   "page_count": page_count,
                   "char_count": len(text),
               }

        4) 返回 Document(...)，doc_id 用 ``stable_id(str(path), text)``。
        """

        try:
            text, page_count = await run_blocking(_extract_pdf_text, path)
        except Exception as exc:
            raise ParseError(f"解析 PDF 失败: {exc}", source=str(path)) from exc

        if not text.strip():
            raise ParseError(
                "PDF 里没有抽出任何文字，可能是扫描件（需要 OCR 才能用）",
                source=str(path),
            )

        metadata = {
            "path": str(path),
            "suffix": ".pdf",
            "size_bytes": path.stat().st_size,
            "page_count": page_count,
            "char_count": len(text),
        }

        return Document(
            doc_id=stable_id(str(path), text),
            text=text,
            source=str(path),
            metadata=metadata,
        )

        # raise NotImplementedError("TODO: run_blocking + 空内容拦截 + 组装 Document")
