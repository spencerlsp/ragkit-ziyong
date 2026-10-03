"""异步文件读写的小工具：这是全项目「同步阻塞 → 异步」的翻译层。

一条原则：
    **同步的活写成普通函数，异步的壳用这两把工具去包。**
    这两把工具就是 asyncio.to_thread 和 aiofiles，M2 之后每个模块都会用到。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import aiofiles

from ..errors import ParseError

__all__ = ["read_text_async", "run_blocking"]

T = TypeVar("T")  # 代表「某一个未知类型，后续再确定」，类似函数参数，但传的是**类型**而不是值。


async def read_text_async(path: Path, encoding: str = "utf-8") -> str:
    """异步读取一个文本文件。

    TODO(你)：用 aiofiles 实现。

    要点：
        1) ``aiofiles.open(...)`` 返回的是**异步上下文管理器**，
           必须写 ``async with``。写成普通的 ``with`` 会直接报 TypeError。
        2) 读内容是 ``await f.read()``（注意 await，不是直接拿返回值）。
        3) 编码**必须**显式写 "utf-8"，原因和 stable_id 那会儿一样：
           Windows 上不指定编码，Python 会用系统默认的 GBK，
           遇到 UTF-8 的中文文件就是乱码或者 UnicodeDecodeError。
        4) 失败要翻译成我们自己的异常：

               try:
                   ...
               except OSError as exc:
                   raise ParseError(f"读取文件失败: {exc}", source=str(path)) from exc

           ``from exc`` 保留原始异常链 —— traceback 里能同时看到
           「ParseError: 读取文件失败」和「FileNotFoundError: ...」两层，根因不丢。
           这是写库的基本素质：**报错的类型由你定，但根因必须留给用户。**

    为什么不用普通的 open()：
        aiofiles 内部其实也是丢给线程池，只是外面套了一层 async 接口，
        让「读文件」能和别的协程并发。你直接写
        ``await asyncio.to_thread(path.read_text, encoding="utf-8")`` 效果接近，
        这里用 aiofiles 是让你见一次主流写法，两种都该认识。
    """
    try:
        async with aiofiles.open(path, encoding=encoding) as f:
            content = await f.read()
            return content
    except OSError as exc:
        raise ParseError(f"读取文件失败: {exc}", source=str(path)) from exc
    # raise NotImplementedError("TODO: 用 aiofiles 异步读文本，失败转成 ParseError")


async def run_blocking(func: Callable[..., T], /, *args: Any) -> T:
    """把一个同步的阻塞函数丢到线程池里跑，返回它的结果。

    TODO(你)：一行搞定它。

    提示：
        - 答案是 ``await asyncio.to_thread(func, *args)``，
          别忘了在文件顶部 ``import asyncio``。
        - 参数里的 ``/`` 表示「func 之前只能按位置传参」，
          这样 ``func`` 这个名字不会和后面的 ``*args`` 打架。

    为什么值得包这一层（而不是到处直接写 to_thread）：
        「哪些调用用了线程池」这个决定从此只有一个地方。
        将来想加超时、或者换成 ProcessPoolExecutor（CPU 密集时会需要，
        因为 Python 有 GIL，to_thread 并不能真正并行算数），只改这一个函数。

    留个印象：``to_thread`` 用的是 loop 的默认线程池，默认上限约
    ``min(32, os.cpu_count() + 4)`` 个线程。M4 你会学到用 Semaphore
    在更上层限制并发，那比调线程池更精准。
    """
    return await asyncio.to_thread(func, *args)
    # raise NotImplementedError("TODO: await asyncio.to_thread(func, *args)")
