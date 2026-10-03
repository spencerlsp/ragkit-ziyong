"""文件解析层：把磁盘上的文件变成 Document。

对外只有两个入口：
    await parse_file(path)            解析单个文件 -> Document
    await parse_files([p1, p2, ...])  并发解析多个文件 -> list[Document]（顺序与输入一致）

再加一个 register(parser) 用来挂自定义解析器。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from ..config import get_settings
from ..errors import ParseError
from ..schemas import Document
from .base import Parser, get_parser, register, supported_extensions
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .text_parser import TextParser

__all__ = ["Parser", "parse_file", "parse_files", "register", "supported_extensions"]

# 注册内置解析器。
# 放在这里（而不是各个解析器模块内部自动注册）是为了「到底装了哪些解析器」一眼可见。
register(TextParser())
register(PdfParser())
register(DocxParser())


async def parse_file(path: str | Path) -> Document:
    """解析单个文件。

    TODO(你)：四步。

        1) ``p = Path(path)``
           —— 收 str 也收 Path，让调用方不用关心我们内部怎么存。

        2) 先检查存在性：

               if not p.is_file():
                   raise ParseError(f"文件不存在或不是普通文件: {p}", source=str(path))

           为什么不依赖底层报错：直接让 pypdf 去读不存在的文件，它会抛
           FileNotFoundError，用户写 ``except ParseError`` 就兜不住。
           在入口处统一成我们自己的异常类型，调用方只需要记一种异常。

           注意 ``is_file()`` 也是同步调用，但它是**本地元数据查询**，
           耗时以微秒计，不值得包 to_thread。这是工程判断：
           **不是所有同步调用都要异步化，只有真正会阻塞的才需要。**

        3) ``parser = get_parser(p)``   # 找不到会自己抛 ParseError

        4) ``return await parser.parse(p)``

           —— 注意是 return await，不是 return。忘了 await 你会得到一个协程对象，
           而且这个错误非常隐蔽：类型检查器可能不报、代码也不炸，
           直到你在别处对返回值取 ``.text`` 才报「coroutine has no attribute」。
           这类 bug 叫「忘了 await」，是 asyncio 新手第一大坑。
    """
    p = Path(path)

    if not p.is_file():
        raise ParseError(f"文件不存在或不是普通文件: {p}", source=str(path))

    parser = get_parser(p)

    return await parser.parse(p)

    # raise NotImplementedError("TODO: 校验文件 -> 查解析器 -> await 调用")


async def parse_files(
    paths: Iterable[str | Path],
    *,
    max_concurrency: int | None = None,
) -> list[Document]:
    """并发解析多个文件。

    TODO(你)：用 asyncio.TaskGroup + asyncio.Semaphore 实现。
    骨架如下，把注释逐条翻译成真正能跑的代码。

        items = [Path(p) for p in paths]
        # ↑ 先把可迭代对象「固化」成 list：下面要用两次
        #   （一次算长度、一次遍历），而生成器用过一次就空了。

        results: list[Document | None] = [None] * len(items)

        limit = max_concurrency or get_settings().max_concurrency
        sem = asyncio.Semaphore(limit)

        async def one(index: int, path: Path) -> None:
            async with sem:                     # ← 并发闸门，必须在协程内部
                results[index] = await parse_file(path)

        async with asyncio.TaskGroup() as tg:
            for i, p in enumerate(items):
                tg.create_task(one(i, p))

        # 走出 async with 时，TaskGroup 保证所有任务都已结束
        return [doc for doc in results if doc is not None]

    三个必须想明白的点：

      1) **为什么要 sem（信号量）**
         TaskGroup 会**无限制**地创建任务。解析 5000 个 PDF 等于一次性
         往线程池塞 5000 个任务排队，内存和文件句柄都会爆。
         ``Semaphore(limit)`` 保证「同时真正在跑」的任务不超过 limit 个，
         其余的在 ``async with sem`` 这一行安静排队。
         关键：闸门必须在**协程内部**（``async with sem:``），
         不能包在 ``tg.create_task(...)`` 外面 —— 包外面就变成串行了，
         你会得到一个正确但慢得离谱的实现。

      2) **为什么用 results[index] 回填**
         TaskGroup 完成任务**不保证顺序**：谁先解析完谁先写。
         如果直接 ``results.append(...)``，输出顺序会随文件大小随机变化 ——
         一个 10KB 的 txt 可能排在 50MB 的 PDF 前面。
         用下标回填是保证「输出顺序 == 输入顺序」最简单的办法。

      3) **为什么最后要过滤掉 None**
         ``results`` 声明成 ``list[Document | None]`` 是给 mypy 看的
         （它不知道任务一定填满了所有槽位）。返回时过滤掉 None，
         既给 mypy 一个交代，也让类型变成 list[Document]。
         注意：如果有任务抛异常，TaskGroup 会在这里抛 ExceptionGroup，
         根本走不到 return —— 所以过滤 None 不是用来掩盖错误的。

    补充：异常行为
        任何一个文件解析失败，都会让整个 TaskGroup 退出，其余正在跑的任务
        **会被自动取消**，最后抛出一个 ExceptionGroup，里面包着所有失败的异常
        （用 ``except* ParseError`` 可以分别捕获）。
        这比老的 asyncio.gather 安全得多：gather 默认不取消兄弟任务，
        会有任务变成没人管的野协程。
        真实项目里你可能想「跳过坏文件、继续处理好的」，那就在 one() 里
        try/except 收集错误 —— 留到 M8 评估时再做。现在先让错误直接暴露，
        暴露得越吵越好。
    """
    items = [Path(p) for p in paths]

    results: list[Document | None] = [None] * len(items)

    limit = max_concurrency or get_settings().max_concurrency
    sem = asyncio.Semaphore(limit)

    async def one(index: int, path: Path) -> None:
        async with sem:  # ← 并发闸门，必须在协程内部
            results[index] = await parse_file(path)

    async with asyncio.TaskGroup() as tg:
        for i, p in enumerate(items):
            tg.create_task(one(i, p))

    return [doc for doc in results if doc is not None]
    # raise NotImplementedError("TODO: TaskGroup + Semaphore，保持输出顺序")
