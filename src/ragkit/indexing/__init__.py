"""索引层：把 chunk 和向量写进 Milvus。

对外入口：
    MilvusIndexer        低层 CRUD（确保建表 / 写入 / 删除 / 计数 / 删整表）
    ingest_chunks()      高层流水线（删旧 -> 向量化 -> 写入）
    build_schema()       单独导出，方便测试和排查
    build_index_params()
"""

from __future__ import annotations

from .ingest import ingest_chunks
from .milvus import MilvusIndexer
from .schema import build_index_params, build_schema

__all__ = [
    "MilvusIndexer",
    "ingest_chunks",
    "build_schema",
    "build_index_params",
]
