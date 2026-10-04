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
    """把 chunk 逐行写成 JSONL 文件，返回写入的条数。

    TODO(你)：用 aiofiles 的**写**模式。

        1) 打开写模式：

               async with aiofiles.open(path, "w", encoding="utf-8") as f:

           —— 是 ``async with``，M2 那个坑别踩第二次。
              写模式直接写 "w" 就行，别再加多余的参数。

        2) 逐条写：

               for chunk in chunks:
                   await f.write(chunk.model_dump_json() + "\\n")

           两个细节：
             * 每行**必须自己加** "\\n"。``write()`` 不会替你补换行 ——
               不加的话所有 JSON 会挤成一行，JSONL 就退化成「一行超长 JSON」了。
             * 用 pydantic 的 ``model_dump_json()`` 一行拿到合法 JSON，
               比手写 ``json.dumps(chunk.model_dump())`` 省事，
               还自动处理了 datetime 这类不能直接 json 化的类型。

        3) 返回写入条数 ``len(chunks)``

    为什么用 JSONL 而不是一个大 JSON 数组：
        * 一行一条，用 ``Get-Content chunks.jsonl | Select-Object -First 5``
          就能看开头几条，不用把整个文件读进内存；
        * 可以流式追加、流式读；一行坏了不影响其他行，排查成本低。
        日志、数据集、评测结果都爱用这个格式。
    """
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            await f.write(chunk.model_dump_json() + "\n")
    return len(chunks)
    # raise NotImplementedError("TODO: aiofiles 写模式逐行写 JSONL")
