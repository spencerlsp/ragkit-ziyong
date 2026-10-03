"""解析层的公共约定：一个 Protocol 加一个按扩展名索引的注册表。

设计意图：主流程（parse_file）不应该知道有哪些解析器。
它只做两件事 —— 按后缀查表、调用查到的解析器。
这样你以后要加 .xlsx / .html / .epub，只需要写一个新类然后 register()，
一行主流程代码都不用改。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..errors import ParseError
from ..schemas import Document

__all__ = ["Parser", "register", "get_parser", "supported_extensions"]


# 加上 @runtime_checkable 后，可以写 isinstance(对象, Parser) 做运行时检查
# （只检查有没有同名方法，不检查签名）。
@runtime_checkable
class Parser(Protocol):
    """解析器协议 —— 描述「一个解析器长什么样」，而不是规定「必须继承谁」。

    ★ 这是模仿案例 ★
    任何类只要有一个 ``extensions`` 类属性和一个 ``async def parse`` 方法，
    **不需要继承任何东西**，就算满足这个 Parser。这叫结构化子类型
    （鸭子类型的类型化版本）。

    为什么用 Protocol 而不是抽象基类：
      * 抽象基类要求实现者写 ``class X(MyABC)``，等于把第三方的手绑住；
      * Protocol 只看形状，任何类都能直接「成为一个 Parser」。
    """

    extensions: tuple[str, ...]

    async def parse(self, path: Path) -> Document:
        """把 ``path`` 解析成一个 Document。"""
        ...


# 全局注册表：后缀 -> 解析器实例。模块级可变状态，所以只在 register() 里写它。
_PARSERS: dict[str, Parser] = {}


def register(parser: Parser) -> Parser:
    """把一个解析器登记进注册表，并把解析器本身返回。

    这里有个容易忽略的点：``ext.lower()``。
    不统一小写的话，".PDF" 和 ".pdf" 会变成两个不同的键，用户传 ".PDF" 就查不到了。
    """
    for ext in parser.extensions:
        _PARSERS[ext.lower()] = parser
    return parser


def get_parser(path: Path) -> Parser:
    """按文件后缀查出对应的解析器；查不到就抛 ParseError。

    TODO(你)：实现它。

    步骤：
        1) ``ext = path.suffix.lower()``
        2) ``parser = _PARSERS.get(ext)``
        3) parser 是 None 就抛 ParseError。错误信息要同时说清「你给的是什么」
           和「我支持什么」，否则用户只知道失败、不知道为什么失败：

               f"不支持的文件类型 {ext!r}，当前支持：{', '.join(supported_extensions())}"

           ParseError 支持 ``source=`` 关键字参数，把 ``str(path)`` 传进去，
           报错时才能定位到具体文件。
        4) 返回 parser

    提示：
      - ``{ext!r}`` 里的 ``!r`` 是用 repr() 格式化：没有后缀的文件会显示成
        ``''`` 而不是一片空白。用户传了个没后缀的文件时，这个细节能省他半小时。
      - 别用 try/except KeyError，``dict.get`` 更直接。
    """
    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        msg = f"不支持的文件类型 {ext!r}，当前支持：{', '.join(supported_extensions())}"
        raise ParseError(msg, source=str(path))
    return parser
    # raise NotImplementedError("TODO: 按后缀查表，查不到抛 ParseError")


def supported_extensions() -> tuple[str, ...]:
    """返回所有已注册的后缀，排好序（方便报错信息和文档里展示）。"""
    return tuple(sorted(_PARSERS))
