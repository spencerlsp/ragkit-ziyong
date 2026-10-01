"""ragkit 的核心数据模型。

整条流水线的数据形状就由这个文件定义::

    文件 --解析--> Document --切分--> Chunk --向量化入库--> (Milvus)
    query --检索--> list[ScoredChunk] --生成--> RAGResult

所有跨模块传递的对象都用 pydantic 模型，好处有三个：
  * 类型明确，IDE 能补全，mypy 能查错；
  * 自动做校验和类型转换（"3" 会变成 3）；
  * ``.model_dump()`` / ``.model_dump_json()`` 一行序列化，方便落盘和写进 Milvus。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class Document(BaseModel):
    """一个被完整解析出来的文件。一个文件 = 一个 Document。"""

    # 文档的稳定 ID。由 utils.stable_id(source, text) 生成，绝不要用随机数。
    doc_id: str

    # 解析出来的纯文本全文。
    text: str

    # 文件来源：本地路径或 URL。允许为 None（比如直接从内存里的字符串建文档）。
    source: str | None = None

    # 任意元数据：页数、作者、文件类型、创建时间…… 全部塞这里。
    # 用 default_factory=dict 而不是默认值 {}，是为了避开「所有实例共享同一个字典」
    # 这个经典 Python 陷阱（pydantic 里同样有这个坑）。
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        """校验器：拒绝空文本，并顺手把两端的空白去掉。

        ``@field_validator("text")`` 表示「给 text 赋值之后、实例构造完成之前」调用它。
        ``@classmethod`` 是 pydantic v2 的硬性要求，忘了会直接报 TypeError。
        返回值就是最终存进模型的值 —— 所以返回 ``value.strip()`` 能起到清洗作用。
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Document.text 不能为空")
        return cleaned

    @computed_field  # type: ignore[prop-decorator]
    @property
    def char_count(self) -> int:
        """文本长度。``@computed_field`` 让它同时出现在 ``.model_dump()`` 里。

        用法：``doc.char_count`` 是 int；``doc.model_dump()`` 也会多出一个
        ``"char_count"`` 键（普通 @property 不会出现在 dump 里）。
        """
        return len(self.text)


class Chunk(BaseModel):
    """切分后的一段文本。一个 Document 通常产生很多个 Chunk。"""

    # 稳定 ID：utils.stable_id(doc_id, str(index), text)
    chunk_id: str 

    # 来自哪个文档（检索到之后要能溯源）
    doc_id: str

    # 这一段的内容
    text: str

    # 在文档内的顺序，从 0 开始
    # ge=0 表示必须 >= 0。用 Field 声明约束，一行顶六行 validator，
    # 而且约束会写进 pydantic 生成的 JSON Schema 和错误信息里。
    index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text") # 对单个字段做校验、预处理、值转换(在创建时便运行)
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Chunk.text 不能为空")
        return cleaned

    # @computed_field 是 pydantic 的写法：让这个计算属性也能被 model_dump 序列化。
    # @property 是 Python 原生的：给实例加一个只读属性，用的时候不用加括号。
    # mypy 不认「装饰器叠装饰器」这种形态，所以上面那行必须挂一个 type: ignore。
    @computed_field  # type: ignore[prop-decorator]
    @property
    def char_count(self) -> int:
        return len(self.text)

    
class ScoredChunk(BaseModel):
    """检索结果：一个 Chunk 加上它的相关性分数。"""

    chunk: Chunk
    score: float

    # frozen=True 表示实例创建后不可修改。检索结果在传递过程中被某处
    # 意外改掉是非常难查的 bug，冻结能让它当场报错 —— 这叫「用类型系统防呆」。
    model_config = ConfigDict(frozen=True)

    def preview(self, limit: int = 80) -> str:
        text = self.chunk.text
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

class RAGResult(BaseModel):
    """一次「查询 -> 检索 -> 生成」的最终产物。"""

    query: str # 用户的初始问题
    answer: str # LLM生成的回答：只做检索时给的空字符串
    contexts: list[ScoredChunk] = Field(default_factory=list) # 回答所依据的检索片段
    model: str | None = None # 生成用的模型名，为方便评估时的记录
    metadata: dict[str, Any] = Field(default_factory=dict) # 耗时、token 数等


    @property
    def context_texts(self) -> list[str]:
        return [sc.chunk.text for sc in self.contexts]
