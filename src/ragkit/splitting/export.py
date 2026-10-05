"""把切分结果落盘，方便肉眼检查。

这是 M3 里**唯一真异步**的地方：写文件是 IO，会等待。
顺手和 processing / splitting 的其他部分对比一下，
你就体会到「sync 还是 async」的判断标准落在哪了。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import aiofiles

from ..schemas import Chunk

__all__ = ["write_chunks_jsonl"]


async def write_chunks_jsonl(chunks: Sequence[Chunk], path: Path) -> int:
    """把 chunk 逐行写成 JSONL 文件，返回写入的条数。"""
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            await f.write(chunk.model_dump_json() + "\n")
    return len(chunks)
    # raise NotImplementedError("TODO: aiofiles 写模式逐行写 JSONL")
