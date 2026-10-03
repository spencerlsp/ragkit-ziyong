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
        """读取文本并包装成 Document。

        TODO(你)：三步走。

        1) ``text = await read_text_async(path)``

        2) 组装 metadata（类型随便填，schema 里已声明成 dict[str, Any]）：

               metadata={
                   "path": str(path),
                   "suffix": path.suffix.lower(),
                   "size_bytes": path.stat().st_size,
               }

           这里 ``path.stat()`` 是同步阻塞调用，我故意没让你包 to_thread ——
           读元数据是本地操作、耗时以微秒计，包异步的开销比收益大。
           这是工程判断，不是偷懒：**只有真正会阻塞的调用才值得异步化。**
           （但你要知道它确实不是 async，将来在几千文件的场景下，
           这类小额同步调用累积起来也会有影响。）

        3) 返回

               Document(
                   doc_id=stable_id(str(path), text),
                   text=text,
                   source=str(path),
                   metadata=metadata,
               )

        为什么用 stable_id 而不是 uuid：
            同一个文件只要内容没变，doc_id 就永远相同。M5 往 Milvus 写数据时，
            重复导入同一批文件会**覆盖**而不是堆重复数据 —— 幂等性就靠它。

        边界：Document 的校验器拒绝空文本，所以空文件会抛 ValidationError
        而不是 ParseError。M2 先接受（空的 txt 本来就是用户自己的问题），
        但 PDF/DOCX 那种「可能是扫描件」的情况必须自己拦住，见 pdf_parser.py。
        """
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
