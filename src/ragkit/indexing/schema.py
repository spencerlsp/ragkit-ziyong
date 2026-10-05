"""Milvus collection 的 schema 定义。

schema 是「数据和数据库之间的合同」：字段名、类型、主键、向量维度、度量方式。
**一旦建好，改 schema 就等于重建 collection + 重灌全部数据。**

单独成文件的好处：构建 schema 和 index_params 完全是**客户端行为**，
不需要连数据库就能测试 —— 测试跑得飞快，还不依赖你本地有没有 Milvus。
"""

from __future__ import annotations

from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType
from pymilvus.milvus_client import IndexParams

from ..errors import IndexingError

__all__ = [
    "CHUNK_ID_FIELD",
    "DOC_ID_FIELD",
    "TEXT_FIELD",
    "CHUNK_INDEX_FIELD",
    "VECTOR_FIELD",
    "SPARSE_FIELD",
    "MAX_ID_LENGTH",
    "MAX_TEXT_LENGTH",
    "build_schema",
    "build_index_params",
]

CHUNK_ID_FIELD = "chunk_id"
DOC_ID_FIELD = "doc_id"
TEXT_FIELD = "text"
CHUNK_INDEX_FIELD = "chunk_index"  # 故意不叫 "index"
VECTOR_FIELD = "embedding"
SPARSE_FIELD = "sparse"  # BM25 用的稀疏向量（由服务端根据 text 自动生成）

# BM25 函数的名字。它把 text 字段「映射」成 sparse 字段 ——
# 之所以要单独起个名字，是因为一个 collection 上可以有多个函数。
BM25_FUNCTION_NAME = "bm25_text_to_sparse"

MAX_ID_LENGTH = 64
MAX_TEXT_LENGTH = 65535  # Milvus VARCHAR 的长度上限


def build_schema(dim: int, *, with_sparse: bool = True) -> CollectionSchema:
    """按向量维度构造 collection 的 schema。"""
    # 0) 维度校验
    if dim <= 0:
        raise IndexingError(f"向量维度必须为正数，收到 {dim}")

    # 1) chunk_id 字符串主键
    field_chunk_id = FieldSchema(
        name=CHUNK_ID_FIELD,
        dtype=DataType.VARCHAR,
        max_length=MAX_ID_LENGTH,
        is_primary=True,
        description="chunk 的稳定 ID（stable_id 生成的内容哈希）",
    )

    # 2) doc_id 溯源字段
    field_doc_id = FieldSchema(
        name=DOC_ID_FIELD,
        dtype=DataType.VARCHAR,
        max_length=MAX_ID_LENGTH,
        description="文档唯一ID，用于文档级过滤与删除",
    )

    # 3) text 正文
    # enable_analyzer=True 是 BM25 的前提：要先分词，才能统计词频。
    # 这个参数是 FieldSchema 的「类型参数」（走 **kwargs），不是普通的可选字段。
    field_text = FieldSchema(
        name=TEXT_FIELD,
        dtype=DataType.VARCHAR,
        max_length=MAX_TEXT_LENGTH,
        enable_analyzer=with_sparse,
        description="chunk 原文文本",
    )

    # 4) chunk_index，当前文档内分片序号
    field_chunk_index = FieldSchema(
        name=CHUNK_INDEX_FIELD,
        dtype=DataType.INT64,
        description="该分片在所属文档内的序号",
    )

    # 5) embedding 向量字段
    field_vector = FieldSchema(
        name=VECTOR_FIELD,
        dtype=DataType.FLOAT_VECTOR,
        dim=dim,
        description="chunk 的稠密向量",
    )

    fields = [field_chunk_id, field_doc_id, field_text, field_chunk_index, field_vector]
    functions = []

    if with_sparse:
        # 6) 稀疏向量字段 + BM25 函数
        #
        # 关键理解：这个字段**不需要你写入**。
        # 你写 text，服务端按 BM25 函数自动算出 sparse 并存下来。
        # 所以 upsert 的行里没有它（见 milvus.py 的 upsert_chunks）。
        fields.append(
            FieldSchema(
                name=SPARSE_FIELD,
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="text 经 BM25 自动生成的稀疏向量",
            )
        )
        functions.append(
            Function(
                name=BM25_FUNCTION_NAME,
                function_type=FunctionType.BM25,
                input_field_names=[TEXT_FIELD],
                output_field_names=[SPARSE_FIELD],
            )
        )

    # 7) 组装 schema
    schema = CollectionSchema(
        fields=fields,
        description="ragkit 的 chunk 向量库",
        enable_dynamic_field=True,
        functions=functions,
    )

    return schema


def build_index_params(*, with_sparse: bool = True) -> IndexParams:
    """构造向量索引的参数。

    ⚠️ 稠密索引用的是 **COSINE**（余弦相似度，**越大越相似**）。
    换成 L2 距离的话方向相反，上层的排序、阈值、Top-K 取法全都要跟着反过来，
    而且不会报任何错 —— 度量方向必须一路传导到最上层。
    """
    params = IndexParams()
    params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    if with_sparse:
        # 稀疏向量用倒排索引 + BM25 度量 —— 这是 Milvus 里的固定搭配，
        # 换成别的组合（比如稀疏向量配 COSINE）服务端会拒绝。
        params.add_index(
            field_name=SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
    return params
