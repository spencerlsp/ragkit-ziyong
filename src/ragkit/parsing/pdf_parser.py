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
    """同步地把 PDF 每页文字抽出来，返回 (全文, 页数)。"""
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
        """解析 .pdf 文件。底层异常统一翻译成 ParseError。

        扫描件（整页是图片）抽不出文字，会显式报错而不是产出空 chunk ——
        放它过去的话，最后会在向量库里堆一批空记录，排查成本高得多。
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
