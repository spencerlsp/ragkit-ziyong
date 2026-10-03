"""全局配置：所有可变参数都集中在这里，通过环境变量 / .env 覆盖。

为什么用 pydantic-settings 而不是自己 ``os.environ.get``：
  * 自动从 .env 读，字段名自动映射成 **大写** 环境变量（openai_api_key -> OPENAI_API_KEY）；
  * 自动类型转换（"8" -> 8、"true" -> True）；
  * 配置写错在**启动时**就报错，而不是跑到一半才崩。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """ragkit 的全部配置。

    TODO(你)：把下面的配置项声明出来。

    第一步，加类级别的配置：::

        model_config = SettingsConfigDict(
            env_file=".env",          # 从项目根目录的 .env 读取
            env_file_encoding="utf-8",
            extra="ignore",           # .env 里有无关变量也不报错
        )

    第二步，声明字段（格式 ``字段名: 类型 = 默认值``）::

        # ---- OpenAI 兼容的 API：Embedding 和 LLM 都走它 ----
        openai_base_url: str = "https://api.openai.com/v1"
        openai_api_key: SecretStr = SecretStr("")
        embedding_model: str = "text-embedding-3-small"
        embedding_dim: int = 1536
        llm_model: str = "gpt-4o-mini"
        llm_temperature: float = 0.0

        # ---- Milvus ----
        milvus_uri: str = "http://localhost:19530"
        milvus_token: str = ""
        milvus_collection: str = "ragkit_chunks"
        milvus_batch_size: int = 500        # 一次 upsert 多少条

        # ---- 切分 ----
        chunk_size: int = 800               # 每个 chunk 的目标字符数
        chunk_overlap: int = 120            # 相邻 chunk 的重叠字符数

        # ---- 检索 ----
        top_k: int = 5

        # ---- 运行时 ----
        max_concurrency: int = 8            # M4 用它限制同时在飞的请求数
        request_timeout: float = 60.0
        max_retries: int = 3

    第三步，加校验器，保证 chunk_overlap 不会把 chunk_size 吃掉：:

        @model_validator(mode="after")
        def _check_chunk_params(self) -> "Settings":
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap 必须小于 chunk_size")
            return self

    提示：
        - ``openai_api_key`` 用 SecretStr，这样 print / 写日志时会显示成
          ``**********``，不会把密钥漏到日志里。真要用时调
          ``settings.openai_api_key.get_secret_value()``。
        - 字段名会被 pydantic-settings 自动映射成同名大写环境变量，
          比如 openai_base_url 对应 OPENAI_BASE_URL。
        - 数值字段可以顺手加约束，例如 ``top_k: int = Field(default=5, gt=0)``
          （这行会用到已经 import 进来的 Field，别让它闲置）。
        - ``mode="after"`` 表示「所有字段都赋值完之后再跑」，所以能直接访问
          ``self.chunk_overlap`` / ``self.chunk_size``。
    """

    # TODO(你)：在这里写 model_config、字段和校验器
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- OpenAI 兼容的 API：LLM ----
    openai_chat_base_url: str = "https://api.deepseek.com"
    openai_chat_api_key: SecretStr = SecretStr("")
    openai_chat_model: str = "deepseek-flash"
    llm_temperature: float = 0.0

    # ----OpenAI 兼容的 API：EMBEDDING ----
    openai_embedding_base_url: str = "https://api.siliconflow.cn/v1"
    openai_embedding_api_key: SecretStr = SecretStr("")
    openai_embedding_model: str = "BAAI/bge-m3"
    openai_embedding_dim: int = 1024  # BAAI/bge-m3 的向量维度；M5 建 Milvus collection 要用

    # ---- Milvus ----
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "ragkit_chunks"
    milvus_batch_size: int = 500  # 一次 upsert 多少条

    # ---- 切分 ----
    chunk_size: int = 800  # 每个 chunk 的目标字符数
    chunk_overlap: int = 120  # 相邻 chunk 的重叠字符数

    # ---- 检索 ----
    top_k: int = 5

    # ---- 运行时 ----
    max_concurrency: int = 8  # M4 用它限制同时在飞的请求数
    request_timeout: float = 60.0
    max_retries: int = 3

    @model_validator(mode="after")  # after：所有字段赋完值后才跑，所以能访问 size 和 overlap
    def _check_chunk_params(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的 Settings 实例。

    TODO(你)：一行 ``return Settings()`` 即可。

    为什么套一层函数 + @lru_cache：
      * 每次重新读 .env 很浪费；更重要的是，配置不该在运行过程中变形，
        缓存能保证整个进程自始至终用的是同一套配置；
      * 测试里想换配置，调 ``get_settings.cache_clear()`` 再重新调用就行。
    """
    # 每次重新读 .env 很浪费；更重要的是，配置不该在运行过程中变形，
    # 缓存能保证整个进程自始至终用的是同一套配置；
    # 测试里想换配置，调 ``get_settings.cache_clear()`` 再重新调用就行。
    return Settings()
