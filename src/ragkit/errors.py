"""ragkit 的异常体系。

为什么要自定义异常：使用方可以 ``except ragkit.ParseError`` 精确捕获，
而不是去猜 ``ValueError`` / ``RuntimeError`` 到底是谁抛出来的。
"""

from __future__ import annotations

__all__ = [
    "RagkitError",
    "ParseError",
    "SplitError",
    "EmbeddingError",
    "IndexingError",
    "RetrievalError",
    "GenerationError",
    "EvaluationError",
]


class RagkitError(Exception):
    """ragkit 所有异常的基类。

    TODO(你)：实现 ``__init__``

    签名（照着抄，不要改）：::

        def __init__(self, message: str, *, source: str | None = None) -> None:

    要做三件事：
        1) ``super().__init__(message)``
           这一步最关键 —— 不调用它，``str(err)`` 会是空字符串，日志里啥也看不到。
        2) ``self.message = message``
        3) ``self.source = source``
           source 表示「是谁出的错」：文件路径、Milvus collection 名、模型名……

    细节提示：
        - ``*`` 表示 source 只能按关键字传：``RagkitError("坏了", source="a.txt")``。
          这样以后加参数也不会悄悄改坏已有的位置参数调用。
        - 不需要写 ``__str__``，父类的实现已经够用了。
    """

    def __init__(self, message: str, *, source: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.source = source  # source 表示「是谁出的错」：文件路径、Milvus collection 名、模型名……


class ParseError(RagkitError):
    """文件解析失败（格式不支持、文件损坏、编码解不开等）。"""


class SplitError(RagkitError):
    """文本切分失败（通常是切分参数不合法）。"""


class EmbeddingError(RagkitError):
    """调用 Embedding 接口失败（超时、限流、返回格式不对等）。"""


class IndexingError(RagkitError):
    """Milvus 建表 / 写入 / 删除失败。"""


class RetrievalError(RagkitError):
    """检索失败（collection 不存在、维度不匹配等）。"""


class GenerationError(RagkitError):
    """调用 LLM 生成答案失败。"""


class EvaluationError(RagkitError):
    """评估过程本身出错（数据集格式不对、没有可用样本等）。"""
