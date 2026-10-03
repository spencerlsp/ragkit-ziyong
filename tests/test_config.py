"""Settings 的测试。

TODO(你)：写下面这些测试。

⚠️ 关键词：**测试隔离**。
你本机迟早会有个 .env，里面的值会影响测试结果。所以凡是断言「默认值」或
「环境变量覆盖」的测试，都要显式传 ``Settings(_env_file=None)`` 关掉 .env 读取，
只让当前进程的环境变量生效。下面的提示里已经标了哪里需要。
"""

from __future__ import annotations

import pytest

from ragkit.config import Settings, get_settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认值应该与代码里写的一致。

    TODO(你)：
      - 用 monkeypatch.delenv(..., raising=False) 把可能存在的环境变量删掉，
        至少删掉 MILVUS_COLLECTION（raising=False 表示不存在也不报错）
      - settings = Settings(_env_file=None)
      - 断言 milvus_collection == "ragkit_chunks"、top_k == 5
    """
    # 删除环境变量 MILVUS_COLLECTION；raising=False：变量不存在也不抛异常
    monkeypatch.delenv("MILVUS_COLLECTION", raising=False)
    # 不加载 .env 文件，只使用模型内置默认值
    settings = Settings(_env_file=None)
    # 断言默认值
    assert settings.milvus_collection == "ragkit_chunks"
    assert settings.top_k == 5
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量能覆盖默认值 —— 这是 pydantic-settings 的核心价值。

    TODO(你)：
      monkeypatch.setenv("TOP_K", "3")
      assert Settings(_env_file=None).top_k == 3     # 注意类型是 int 不是 str
    """
    monkeypatch.setenv("TOP_K", "3")
    assert Settings(_env_file=None).top_k == 3
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_chunk_overlap_must_be_smaller(monkeypatch: pytest.MonkeyPatch) -> None:
    """chunk_overlap >= chunk_size 时应该在构造阶段就报错。

    TODO(你)：
      monkeypatch.setenv("CHUNK_SIZE", "100")
      monkeypatch.setenv("CHUNK_OVERLAP", "200")
      with pytest.raises(ValueError): Settings(_env_file=None)
    """
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_api_key_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    """SecretStr 不该在 print / repr 里泄露明文。

    TODO(你)：
      monkeypatch.setenv("OPENAI_CHAT_API_KEY", "sk-super-secret")
      settings = Settings(_env_file=None)
      断言 "sk-super-secret" 不在 repr(settings) 里
      再断言 settings.openai_chat_api_key.get_secret_value() == "sk-super-secret"
    """
    # 设置环境变量
    monkeypatch.setenv("OPENAI_CHAT_API_KEY", "sk-super-secret")
    # 不加载 .env，只读取环境变量
    settings = Settings(_env_file=None)

    # repr 里不能出现明文密钥
    text = repr(settings)
    assert "sk-super-secret" not in text

    # 但是调用 get_secret_value() 可以拿到原始密钥
    assert settings.openai_chat_api_key.get_secret_value() == "sk-super-secret"
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_get_settings_is_cached() -> None:
    """get_settings 应该缓存，返回同一个对象。

    TODO(你)：
      先 get_settings.cache_clear() 清掉缓存（测试之间互不影响）
      断言 get_settings() is get_settings()
      最后再 get_settings.cache_clear() 收尾
    """
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
    # raise NotImplementedError("TODO: 写出这个测试的断言")
