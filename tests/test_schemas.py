"""Document / Chunk / ScoredChunk / RAGResult 的测试。

TODO(你)：写下面这些测试。

写测试的三个通用套路，本项目会反复用到：
  1) 正常路径：构造对象，用 assert 检查字段值 = 预期值；
  2) 非法输入：``with pytest.raises(ValueError):`` 包住构造；
  3) 边界 / 陷阱：例如「两个实例的默认字典不能是同一个对象」。

下面第 4、5 个测试已经写好了，是给你看的**测试模仿案例**，
剩下的照着写，写完把 raise 那行删掉。
"""

from __future__ import annotations

import pytest

from ragkit.schemas import Chunk, Document, RAGResult, ScoredChunk
from ragkit.utils import stable_id


def make_chunk(text: str = "第一段内容") -> Chunk:
    """测试用的小工厂函数：造一个合法的 Chunk，省得每个测试都抄一遍参数。

    顺便示范 chunk_id 该怎么算：内容哈希，不是随机数，更不能写死。
    """
    return Chunk(
        chunk_id=stable_id("d1", "0", text),
        doc_id="d1",
        text=text,
        index=0,
    )


def test_document_strips_text() -> None:
    """Document 应该把 text 两端的空白去掉。"""
    doc = Document(doc_id="d1", text="  你好  ")
    assert doc.text == "你好"


def test_document_rejects_blank_text() -> None:
    """全空白的 text 必须被拒绝，并且抛的是 ValueError（ValidationError 是它的子类）。"""
    with pytest.raises(ValueError):
        Document(doc_id="d1", text="   ")


def test_document_char_count_appears_in_dump() -> None:
    """char_count 是 computed_field，应该出现在 model_dump() 里。

    TODO(你)：写断言
      - doc.char_count == 2
      - "char_count" in doc.model_dump()
    """
    doc = Document(doc_id="d1", text="你好")
    assert doc.char_count == 2
    assert "char_count" in doc.model_dump()
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_document_metadata_is_per_instance() -> None:
    """两个 Document 的 metadata 不能是同一个 dict。

    TODO(你)：
      a = Document(doc_id="a", text="x")
      b = Document(doc_id="b", text="y")
      a.metadata["k"] = 1
      assert "k" not in b.metadata
    """
    a = Document(doc_id="a", text="x")
    b = Document(doc_id="b", text="y")
    a.metadata["k"] = 1
    assert "k" not in b.metadata
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_chunk_rejects_negative_index() -> None:
    """index 用了 Field(ge=0)，所以 -1 应该被拒绝。"""
    with pytest.raises(ValueError):
        Chunk(chunk_id="c1", doc_id="d1", text="x", index=-1)


def test_chunk_id_is_stable() -> None:
    """chunk_id 是内容哈希：同样输入必须算出同样的 id，index 变了 id 也要变。

    注意为什么非得换个参数再比一次：如果 chunk_id 是写死的（比如 "c1"），
    光比「两次相等」照样能过，测试就白写了 —— 这种测试叫「假绿灯」。
    """
    assert make_chunk("x").chunk_id == make_chunk("x").chunk_id

    other = Chunk(chunk_id=stable_id("d1", "1", "x"), doc_id="d1", text="x", index=1)
    assert other.chunk_id != make_chunk("x").chunk_id


def test_scored_chunk_is_frozen() -> None:
    """ScoredChunk 是 frozen 的，改 score 必须报错。

    TODO(你)：写断言。提示：frozen 报的是 pydantic 的 ValidationError，
    它是 ValueError 的子类，所以 ``pytest.raises(ValueError)`` 能捕获到。
    """

    sc = ScoredChunk(chunk=make_chunk(), score=0.8)
    with pytest.raises(ValueError):
        sc.score = 0.9
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_scored_chunk_preview_truncates() -> None:
    """preview 在文本过长时要截断并加省略号。

    TODO(你)：
      - 短文本：preview() 原样返回，不带 "..."
      - 长文本：preview(limit=3) 的结果以 "..." 结尾
    """
    # 显式构造：文本长度 5，可控，不依赖 make_chunk
    chunk = make_chunk(text="abcde")
    sc = ScoredChunk(chunk=chunk, score=1.0)

    # limit=2：文本长度5>2 → 截断，带 ...
    preview_short = sc.preview(limit=2)
    assert "..." in preview_short
    assert preview_short.startswith("ab")

    # limit=10：5 <= 10 → 不截断，没有 ...
    preview_long = sc.preview(limit=10)
    assert "..." not in preview_long
    assert preview_long == "abcde"

    # limit=5：长度正好等于 limit → 依旧不算截断，不能加 ...
    assert sc.preview(limit=5) == "abcde"


def test_rag_result_defaults() -> None:
    """RAGResult 不传 contexts 时应该是空列表，context_texts 也是空列表。

    TODO(你)：写断言，并确认它没和「只做检索」的用法冲突
    （answer 允许是空字符串）。
    """
    rr = RAGResult(
        query="",
        answer="",
    )

    assert not rr.answer and  rr.context_texts == []
    # raise NotImplementedError("TODO: 写出这个测试的断言")


def test_rag_result_context_texts() -> None:
    """context_texts 应该按顺序取出所有 chunk 的文本。"""
    result = RAGResult(
        query="q",
        answer="a",
        contexts=[ScoredChunk(chunk=make_chunk("x"), score=0.9)],
    )
    assert result.context_texts == ["x"]
