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
    """异步读取一个文本文件。"""
    try:
        async with aiofiles.open(path, encoding=encoding) as f:
            content = await f.read()
            return content
    except OSError as exc:
        raise ParseError(f"读取文件失败: {exc}", source=str(path)) from exc
    # raise NotImplementedError("TODO: 用 aiofiles 异步读文本，失败转成 ParseError")


async def run_blocking(func: Callable[..., T], /, *args: Any) -> T:
    """把一个同步的阻塞函数丢到线程池里跑，返回它的结果。"""
    return await asyncio.to_thread(func, *args)
    # raise NotImplementedError("TODO: await asyncio.to_thread(func, *args)")
