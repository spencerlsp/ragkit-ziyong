"""切分器的测试。

同样注意转义：段落之间是 "\\n\\n"（源码里一个反斜杠），不是 "\\\\n\\\\n"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.errors import SplitError
from ragkit.schemas import Document
from ragkit.splitting import (
    FixedSplitter,
    RecursiveSplitter,
    split_document,
    split_documents,
    write_chunks_jsonl,
)
from ragkit.splitting.recursive import _split_recursive


def make_document(text: str, doc_id: str = "d1") -> Document:
    """小工厂。已经写好了，直接拿去用。"""
    return Document(doc_id=doc_id, text=text, source=f"{doc_id}.txt")


# 前几条测试共用的长文本。10 句话、60 个字符。
# 为什么要专门抽出来：原来那几句只有 24 个字符，配 chunk_size=30/50
# 只会切出 1 个 chunk —— 那种情况下「index 连续」「id 不重复」
# 都是废话式的通过，测试等于没写。
LONG_TEXT = (
    "第一句内容。第二句内容。第三句内容。第四句内容。第五句内容。"
    "第六句内容。第七句内容。第八句内容。第九句内容。第十句内容。"
)


def test_split_returns_sequential_indexes() -> None:
    """TODO(你)：切一段长文本，断言返回的 Chunk 的 index 是 0,1,2,... 连续的。"""
    splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=5)
    doc = make_document(LONG_TEXT)
    chunks = splitter.split(doc)
    assert len(chunks) >= 2, "文本太短只会切出 1 块，这条测试就失去意义了"
    indexes = [c.index for c in chunks]
    assert indexes == list(range(len(chunks))), f"index 不连续，得到 {indexes}"


def test_chunk_ids_are_unique_and_deterministic() -> None:
    """TODO(你)：切两次，断言两次的 chunk_id 列表完全相同；
    并且同一批里的 chunk_id 互不重复（没写错 index、没漏掉 text 参与哈希）。
    """
    splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=5)
    doc = make_document(LONG_TEXT)
    # 第一次切分
    chunks1 = splitter.split(doc)
    ids1 = [c.chunk_id for c in chunks1]
    # 第二次切分，输入完全一样
    chunks2 = splitter.split(doc)
    ids2 = [c.chunk_id for c in chunks2]

    # 1. 两次切分得到完全相同的chunk_id（确定性）
    assert ids1 == ids2, "多次切分相同文档，chunk_id不一致，stable_id失效"
    # 2. 同一次切分内部所有chunk_id互不重复
    assert len(ids1) == len(set(ids1)), f"同次切分出现重复chunk_id: {ids1}"
    assert len(ids1) >= 2, "只有一个 chunk 的话，「不重复」是废话"


def test_chunks_respect_chunk_size_without_overlap() -> None:
    """TODO(你)：用 RecursiveSplitter(chunk_size=50, chunk_overlap=0)，
    断言每个 chunk 的长度 <= 50。

    注意为什么必须 overlap=0：有 overlap 时每个 chunk 会比 chunk_size
    多出最多 overlap 个字符（上一块的尾巴被接在了开头）。
    这是刻意设计，不是 bug —— 但也别图省事把断言放宽成
    ``<= chunk_size + overlap`` 就完事，那样你就失去了对 chunk_size 的控制感。
    overlap 的效果另有一条测试专门验证。
    """
    splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
    doc = make_document(LONG_TEXT)
    chunks = splitter.split(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= 50, f"chunk超长，长度={len(chunk.text)},文本={repr(chunk.text)}"


def test_overlap_shares_text_between_neighbors() -> None:
    """TODO(你)：chunk_overlap=10 时，断言后一个 chunk 的开头
    和前一个 chunk 的结尾有公共部分：

        assert chunks[1].text[:10] == chunks[0].text[-10:]

    用一段**没有空白**的长文本更稳（有空白时末尾可能正好是空格，
    断言会莫名其妙失败）。
    """
    splitter = RecursiveSplitter(chunk_size=40, chunk_overlap=10)
    doc = make_document("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3)
    chunks = splitter.split(doc)
    assert len(chunks) >= 2
    assert chunks[1].text[:10] == chunks[0].text[-10:]


def test_recursive_prefers_paragraph_boundary() -> None:
    """TODO(你)：造一段「两个段落」的文本，chunk_size 刚好装得下一段装不下两段，
    断言切出来的第一块正好是第一个段落（结尾停在段落边界）。

    这是递归切分存在的**唯一理由**：让 chunk 尽量落在自然边界上。
    如果这条不成立，那 FixedSplitter 就够用了，我们也不必写这个模块。
    """
    para1 = "这是第一段完整的段落内容。"
    para2 = "这是第二段完整的段落内容。"
    text = f"{para1}\n\n{para2}"
    splitter = RecursiveSplitter(chunk_size=len(para1) + 5, chunk_overlap=0, separators=["\n\n"])
    doc = make_document(text)
    chunks = splitter.split(doc)
    assert chunks[0].text == para1


def test_pieces_reassemble_to_original() -> None:
    """TODO(你)：断言 _split_recursive 的产物拼回去等于原文。

        separators = ["\\n\\n", "\\n", " "]
        pieces = _split_recursive(text, separators, chunk_size=20)
        assert "".join(pieces) == text

    这是切分器的**不变量**：切分只允许「切开」，不允许「丢字」或「多字」。
    一旦以后有人在 _split_recursive 里加逻辑时忘了把分隔符粘回去，
    这条测试会立刻红。

    为什么敢测一个下划线开头的私有函数：测私有函数通常不好
    （会把实现细节锁死），但这条测的是**数学性质**而不是实现细节，值得。
    """
    text = "hello world\nanother line\n\nnew paragraph"
    separators = ["\n\n", "\n", " "]
    pieces = _split_recursive(text, separators, chunk_size=20)
    assert "".join(pieces) == text


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),  # 这一条是防死循环的：overlap == size 会让切分原地打转
    ],
)
def test_invalid_params_raise_split_error(chunk_size: int, chunk_overlap: int) -> None:
    """参数不合法必须在**构造时**就抛 SplitError，而不是等到切分时才炸。

    @pytest.mark.parametrize 必须挂在**被 pytest 收集的测试函数**上。
    挂在嵌套的辅助函数上是完全无效的（pytest 只收集模块级的 test_* 函数），
    而且那个嵌套函数没有类型注解，mypy 会直接报错。
    """
    with pytest.raises(SplitError):
        RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_fixed_splitter_covers_the_whole_text() -> None:
    """TODO(你)：FixedSplitter(chunk_size=10, chunk_overlap=3) 切一段 100 字的文本，
    断言首尾都被覆盖到：整段文本的开头出现在第一个 chunk 里，
    整段文本的结尾出现在最后一个 chunk 里。

    这条在防「切片时把最后一小截漏掉」——range + 切片最容易出这个错。
    """
    text = "0123456789" * 10  # 刚好100字符
    splitter = FixedSplitter(chunk_size=10, chunk_overlap=3)
    doc = make_document(text)
    chunks = splitter.split(doc)
    # step = 10 - 3 = 7，起点是 0,7,14,...,98，一共 15 块
    assert len(chunks) == 15
    # 第一块必须**正好**是开头 10 个字符。
    # 只写 text.startswith(chunks[0].text) 太松了：哪怕第一块只有 1 个字符也照样通过。
    assert chunks[0].text == text[:10]
    # 最后一块必须一直盖到文本结尾，且非空（防「range + 切片漏掉最后一截」）
    assert chunks[-1].text
    assert text.endswith(chunks[-1].text)


def test_split_documents_returns_one_group_per_document() -> None:
    """TODO(你)：传两个不同的 Document，断言：
    - 返回的列表长度是 2
    - 每一组的 chunk.doc_id 都等于对应文档的 doc_id
    """
    doc1 = make_document("文档A内容", doc_id="doc-a")
    doc2 = make_document("文档B内容", doc_id="doc-b")
    splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
    # 必须真的调用 split_documents，不能用列表推导自己拼一个同名结果 ——
    # 那样测的是列表推导，不是被测函数（典型假绿灯）
    batch_result = split_documents([doc1, doc2], splitter=splitter)
    assert len(batch_result) == 2
    group1, group2 = batch_result
    assert all(c.doc_id == "doc-a" for c in group1)
    assert all(c.doc_id == "doc-b" for c in group2)


def test_split_document_delegates_to_splitter() -> None:
    """split_document 应该把 document 原样交给传进来的 splitter。

    这条让 split_document 这个入口也被真正覆盖到（顺带解决 ruff 的 F401）。
    """
    splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
    doc = make_document(LONG_TEXT)
    assert split_document(doc, splitter=splitter) == splitter.split(doc)


async def test_write_chunks_jsonl_roundtrip(tmp_path: Path) -> None:
    """TODO(你)：切一段文本，写到 tmp_path / "chunks.jsonl"，再读回来验证：

      - 返回值 == chunk 数量
      - 用 ``target.read_text(encoding="utf-8").splitlines()`` 得到的行数
        正好等于 chunk 数量（多一个空行就说明你多写了换行，少一行说明漏写了）
      - 每行 ``json.loads`` 都能解析，且 chunk_id 的集合对得上

    读回来用同步的 Path.read_text 就行 —— 测试代码里同步读没问题，
    被测试的**生产代码**才是异步的。别把测试自己也搞复杂了。

    提示：需要 ``import json``，写在测试函数内部也可以（局部 import 是合法的）。
    """
    import json

    splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=0)
    doc = make_document("测试jsonl序列化，第一句。第二句。第三句。")
    chunks = splitter.split(doc)
    target = tmp_path / "chunks.jsonl"

    written_count = await write_chunks_jsonl(chunks, target)

    assert written_count == len(chunks)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)

    loaded_ids = set()
    for line in lines:
        obj = json.loads(line)
        loaded_ids.add(obj["chunk_id"])

    original_ids = {c.chunk_id for c in chunks}
    assert loaded_ids == original_ids
