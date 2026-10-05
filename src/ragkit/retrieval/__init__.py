"""检索层：把查询文本变成一批带分数的片段。

对外入口：
    Retriever      组合 Embedder + MilvusIndexer（推荐用它，生命周期清晰）
    doc_filter()   构造「只在这些文档里检索」的 filter 表达式

为什么要有这一层，而不是让调用方自己写「embed 再 search」：
    * 两段之间需要共享一个**总超时**（横跨两个 await）；
    * 两个组件的生命周期需要统一管理；
    * 阈值过滤、Top-K 默认值这些约定要收在一处。
    这三件事都不该出现在每个调用点。
"""

from __future__ import annotations

from .reranker import HttpReranker, Reranker, create_reranker
from .retriever import Retriever, doc_filter

__all__ = [
    "Retriever",
    "doc_filter",
    "Reranker",
    "HttpReranker",
    "create_reranker",
]
