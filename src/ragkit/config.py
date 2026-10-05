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
    """ragkit 的全部配置。"""

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
    openai_embedding_batch_size: int = 32  # 一次请求塞多少条文本（服务端一般有上限）

    # ---- 重排（rerank）----
    #
    # 单独一组配置，不复用 openai_embedding_* —— 因为 rerank **不是**
    # OpenAI 兼容协议的一部分（OpenAI 至今没有 rerank 端点）。
    # 它更像各家厂商的「方言」：字段名、返回结构都可能有差异。
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: SecretStr = SecretStr("")
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---- Milvus ----
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_db: str = ""  # 空 = 用 Milvus 的 default 库；库必须先存在，Milvus 不会自动建
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
    retry_base_delay: float = 1.0  # 指数退避的基数：1s、2s、4s……

    @model_validator(mode="after")  # after：所有字段赋完值后才跑，所以能访问 size 和 overlap
    def _check_chunk_params(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的 Settings 实例。"""
    # 每次重新读 .env 很浪费；更重要的是，配置不该在运行过程中变形，
    # 缓存能保证整个进程自始至终用的是同一套配置；
    # 测试里想换配置，调 ``get_settings.cache_clear()`` 再重新调用就行。
    return Settings()
