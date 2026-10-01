"""stable_id 的测试。

TODO(你)：按下面的清单把断言写出来。

规则回顾：
  * 函数名以 ``test_`` 开头，pytest 才会收集它；
  * 一个测试只验证一件事，挂掉时一眼能看出是哪个行为坏了；
  * 断言「这里应该抛错」用 ``with pytest.raises(异常类型):``。

提示：每个测试函数体现在都是 ``raise NotImplementedError``，
所以 ``uv run pytest`` 会全红 —— 这是故意的（TDD 的红灯阶段）。
"""

from __future__ import annotations

import pytest

from ragkit.utils import stable_id


def test_stable_id_is_deterministic() -> None:
    """同一个输入调用两次，结果必须完全相等。"""
    assert stable_id("a.txt", "hello") == stable_id("a.txt", "hello")
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_stable_id_length() -> None:
    """默认长度是 16；显式传 length=8 时长度是 8。"""
    assert len(stable_id("x", length=8)) == 8
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_stable_id_rejects_non_positive_length() -> None:
    """length <= 0 应该抛 ValueError。"""
    with pytest.raises(ValueError) as exc_info:
        stable_id("", length=-1)

    assert "length" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        stable_id("", length=0)
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_stable_id_separates_parts() -> None:
    """("ab", "c") 和 ("a", "bc") 必须算出不同的 ID（验证分隔符没偷懒）。"""
    assert stable_id("ab", "c") != stable_id("a", "bc")
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_stable_id_handles_unicode() -> None:
    """中文输入不能报错（验证 utf-8 编码写对了），长度也要对。"""
    assert stable_id("中文输入")
    # raise NotImplementedError("TODO: 写出这个测试的断言")
