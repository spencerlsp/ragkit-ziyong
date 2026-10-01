"""通用小工具：与业务无关、可独立测试的纯函数。

设计原则：这里的函数 **不依赖网络、不依赖数据库**，
输入输出都是普通 Python 对象，所以它们的测试写起来又快又死板（不需要 mock）。
"""

from __future__ import annotations

import hashlib

__all__ = ["stable_id"]


def stable_id(*parts: str, length: int = 16) -> str:
    """把任意几段字符串折叠成一个 **确定性** 的短 ID。

    为什么需要它：
        RAG 里同一份文件会被反复导入、同一段文本会被重复切分出来。
        如果 ID 每次都随机（比如 uuid4），Milvus 里就会堆出一份份重复数据。
        用「内容的哈希」当 ID，就能做到「重复导入 = 覆盖旧数据」（幂等）。

    参数:
        *parts: 参与哈希的字符串片段，例如 ("data/a.txt", "第一段内容")。
        length: 返回多少个十六进制字符。16 个字符 = 64 bit，冲突概率极低且够短。

    返回:
        全小写的十六进制字符串，长度恰好等于 ``length``。

    实现提示（按顺序照做）：
        1) 校验参数：``length`` 必须为正数，否则 ``raise ValueError``。
           写成 ``if length <= 0: raise ValueError("length 必须为正数")``。
        2) 拼接：用 "\\x1f"（ASCII 的 Unit Separator 控制字符）当分隔符。
           ``payload = "\\x1f".join(parts)``
           为什么必须加分隔符：("ab", "c") 和 ("a", "bc") 直接拼都是 "abc"，
           加了分隔符就变成 "ab\\x1fc" 和 "a\\x1fbc"，不会撞车。
        3) 哈希：``hashlib.sha256(payload.encode("utf-8")).hexdigest()``
           必须显式写 encoding="utf-8"，否则 Windows 默认用 GBK，
           同一段中文在别的机器上算出来的 ID 就不一样了（跨机器就不幂等了）。
        4) 返回前 ``length`` 个字符（切片 ``[:length]``）。

    TODO(你)：把上面第 1~4 步翻译成代码，然后删掉最后那行 raise。

    写完自测（粘到 ``uv run python`` 里跑，或直接跑 tests/test_utils.py）：
        >>> stable_id("a.txt", "hello") == stable_id("a.txt", "hello")
        True
        >>> stable_id("ab", "c") != stable_id("a", "bc")
        True
        >>> len(stable_id("x", length=8))
        8
    """
    if length <= 0:
        raise ValueError("length 必须为正数")
    # 注意这里是**一个**控制字符：源码里写 "\x1f"。
    # 文档字符串里显示的 "\\x1f" 是文档层的转义，别照着文档抄进代码。
    payload = "\x1f".join(parts)
    hash_code = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hash_code[:length]
