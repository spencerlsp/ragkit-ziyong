"""Milvus collection 的 schema 定义。

schema 是「数据和数据库之间的合同」：字段名、类型、主键、向量维度、度量方式。
**一旦建好，改 schema 就等于重建 collection + 重灌全部数据。**

单独成文件的好处：构建 schema 和 index_params 完全是**客户端行为**，
不需要连数据库就能测试 —— 测试跑得飞快，还不依赖你本地有没有 Milvus。
"""

from __future__ import annotations

from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.milvus_client import IndexParams

from ..errors import IndexingError

__all__ = [
    "CHUNK_ID_FIELD",
    "DOC_ID_FIELD",
    "TEXT_FIELD",
    "CHUNK_INDEX_FIELD",
    "VECTOR_FIELD",
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

MAX_ID_LENGTH = 64
MAX_TEXT_LENGTH = 65535  # Milvus VARCHAR 的长度上限


def build_schema(dim: int) -> CollectionSchema:
    """按向量维度构造 collection 的 schema。

    TODO(你)：定义 6 个字段，然后组装。

        0) 先校验维度：

               if dim <= 0:
                   raise IndexingError(f"向量维度必须为正数，收到 {dim}")

           这是第三道防线了（M1 的 Settings、M4 的客户端、这里）。
           为什么值得三处都拦：到这里离数据真正落库只剩一步。
           维度为 0 的 schema 要么被服务端拒绝，要么更糟 —— 建成了但写不进数据。

        1) 主键（**字符串**主键，不是自增整数）：

               FieldSchema(
                   name=CHUNK_ID_FIELD,
                   dtype=DataType.VARCHAR,
                   max_length=MAX_ID_LENGTH,
                   is_primary=True,
                   description="chunk 的稳定 ID（stable_id 生成的内容哈希）",
               )

           为什么用字符串主键而不是自增 id：
           我们的 chunk_id 是内容哈希，天然幂等 —— 同一份数据重复导入，
           upsert 会**覆盖同一行**而不是新增一行。自增 id 做不到这点，
           你得自己维护「哪条记录对应哪个 chunk」的映射表。

        2) 溯源字段 doc_id：VARCHAR + max_length=MAX_ID_LENGTH。
           —— M6 用它做「只在这篇文档里检索」，也是我们「先按 doc 删除」的依据。

        3) 正文字段 text：VARCHAR + max_length=MAX_TEXT_LENGTH。
           ⚠️ Milvus 的 VARCHAR 有长度上限。我们的 chunk_size 默认 800，
              离上限很远；但如果有人把它调到 10 万，会在**写入时**才报错。
              用模块常量把上限显式写出来，比散落魔数强。

        4) 序号字段 chunk_index：INT64。
           —— 注意**别叫 "index"**：这个词在 Milvus / SQL 里到处都是
              （索引、filter 表达式、保留字），多打几个字母能省掉一堆疑惑。

        5) 向量字段：

               FieldSchema(
                   name=VECTOR_FIELD,
                   dtype=DataType.FLOAT_VECTOR,
                   dim=dim,
                   description="chunk 的稠密向量",
               )

        6) 组装：

               CollectionSchema(
                   fields=[...按上面顺序...],
                   description="ragkit 的 chunk 向量库",
                   enable_dynamic_field=True,
               )

           **enable_dynamic_field=True 是什么**：允许写入 schema 里没声明的字段。
           Chunk.metadata 是个自由字典（source、page_count……），
           与其为每个可能出现的键都开一列，不如开后门存成动态字段。
           代价是这些字段没有类型保证、过滤时性能也差一些，
           海量数据时要掂量。（我们只用它存一个 source。）

    提示：FieldSchema 的 `max_length` / `dim` 是通过 **kwargs 传进去的类型参数，
    不是命名参数 —— 照上面写就行。
    """
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
    field_text = FieldSchema(
        name=TEXT_FIELD,
        dtype=DataType.VARCHAR,
        max_length=MAX_TEXT_LENGTH,
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

    # 6) 组装 schema
    schema = CollectionSchema(
        fields=[
            field_chunk_id,
            field_doc_id,
            field_text,
            field_chunk_index,
            field_vector,
        ],
        description="ragkit 的 chunk 向量库",
        enable_dynamic_field=True,
    )

    return schema


def build_index_params() -> IndexParams:
    """构造向量索引的参数。

    TODO(你)：三行。

        params = IndexParams()
        params.add_index(
            field_name=VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        return params

    两个选择值得说清楚：

      * **AUTOINDEX**：让 Milvus 自己挑合适的索引结构，standalone 和云上都支持。
        想手动调优可以换 "HNSW"（配 M / efConstruction）或 "IVF_FLAT"。
        先用 AUTOINDEX —— **不要在没有真实性能数据之前优化索引**。

      * **COSINE**（余弦相似度）：值越大越相似。
        因为 M4 的 HashingEmbedder 和 bge-m3 输出的都是归一化向量，
        余弦和内积在这里几乎等价，选 COSINE 是因为它语义最直观 ——
        M6 里「score 越大越相关」这条约定就来自这里。

        ⚠️ 另一个常见选项是 **L2 距离，它是越小越相似**，方向和 COSINE 正好相反。
        混用是 RAG 里最经典的 bug 来源之一：检索结果的顺序会整体反过来，
        而且不会报任何错。**度量方式必须一路传导到上层的排序逻辑。**
    """
    params = IndexParams()
    params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    return params
