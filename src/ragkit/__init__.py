"""ragkit —— 通用 RAG 工具包。

对外暴露的公共 API 都在这里 re-export，使用者只需要 ``import ragkit``。

TODO(你)：把 ragkit 的公共接口挂出来。

要求：
    1) 从 ``.schemas`` 导入 Document / Chunk / ScoredChunk / RAGResult
    2) 从 ``.config`` 导入 Settings / get_settings
    3) 从 ``.errors`` 导入 RagkitError / ParseError / SplitError /
       EmbeddingError / IndexingError / RetrievalError / GenerationError
    4) 定义 ``__all__ = [...]``，把上面所有名字列进去
    5) 删掉 uv 生成的 ``hello()``（已经删了，不用管）

为什么一定要写 ``__all__``：
    * 它是「这个包对外承诺的接口清单」——写进去的名字才算稳定 API，
      以后重构内部实现不会伤到使用者；
    * 没有 __all__ 时 ``from ragkit import *`` 会把所有不以 _ 开头的名字都导出去，
      包括 import 进来的第三方模块名，很容易污染命名空间。

注意：这里 import 的东西必须真的存在，否则 ``import ragkit`` 直接炸。
后面每加一个模块（parsing / splitting / indexing ...），都回来这里补一行。
"""
from .config import Settings, get_settings
from .errors import (
    EmbeddingError,
    GenerationError,
    IndexingError,
    ParseError,
    RagkitError,
    RetrievalError,
    SplitError,
)
from .schemas import Chunk, Document, RAGResult, ScoredChunk

__all__ = [
    "Document",
    "Chunk",
    "ScoredChunk",
    "RAGResult",
    "Settings",
    "get_settings",
    "RagkitError",
    "ParseError",
    "SplitError",
    "EmbeddingError",
    "IndexingError",
    "RetrievalError",
    "GenerationError",
]