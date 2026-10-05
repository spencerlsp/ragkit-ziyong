"""Embedding 层：把文本变成向量。

对外入口：
    create_embedder()                 按配置造一个 Embedder
    embed_all(embedder, texts, ...)   批量向量化（分片 + 并发 + 保序）

两种实现：
    OpenAICompatEmbedder   真的发 HTTP 请求（生产用）
    HashingEmbedder        纯本地的确定性假向量（测试、离线 demo 用）

M4 的核心是异步。这一层是整个项目里唯一**必须**异步的地方 ——
  它要等网络。你会在这里第一次手写异步 HTTP 客户端，并把它包进
  Semaphore / gather / wait_for / 指数退避重试。
"""

from __future__ import annotations

from .base import Embedder, embed_all
from .hashing import HashingEmbedder
from .openai_compat import OpenAICompatEmbedder, create_embedder

__all__ = [
    "Embedder",
    "embed_all",
    "create_embedder",
    "OpenAICompatEmbedder",
    "HashingEmbedder",
]
