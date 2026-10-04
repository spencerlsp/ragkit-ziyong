"""清洗函数的测试。

⚠️ 写断言时注意转义：想表示「回车换行」要写 "\\r\\n"（源码里一个反斜杠），
   写成 "\\\\r\\\\n" 就变成两个字符的反斜杠 r 了。这个坑你已经踩过两次。
"""

from __future__ import annotations

import pytest

from ragkit.processing import clean_document, normalize_whitespace
from ragkit.schemas import Document


def make_document(text: str) -> Document:
    """小工厂。已经写好了，直接拿去用。"""
    return Document(doc_id="d1", text=text, source="d1.txt")


def test_normalize_unifies_newlines() -> None:
    """TODO(你)：断言 "a\\r\\nb\\rc" 被统一成 "a\\nb\\nc"。"""
    raw = "a\r\nb\rc"
    result = normalize_whitespace(raw)
    assert result == "a\nb\nc"


def test_normalize_drops_zero_width_chars() -> None:
    """TODO(你)：断言 "你\\u200b好\\ufeff" 变成 "你好"。

    验证方式可以狠一点：断言两个字符串**确实相等**，
    而不是只断言 len 变小了 —— 后者太松，抓不住"删错了字符"。
    """
    # ⚠️ 单个反斜杠！写成 "\\u200b" 就变成字面量反斜杠+u+2+0+0+b 六个字符了
    raw = "你\u200b好\ufeff"
    result = normalize_whitespace(raw)
    assert result == "你好"


def test_normalize_strips_line_trailing_spaces() -> None:
    """TODO(你)：断言每一行末尾的空格都被去掉。"""
    raw = "第一行   \n第二行  \n第三行"
    result = normalize_whitespace(raw)
    assert result == "第一行\n第二行\n第三行"


def test_normalize_collapses_blank_lines() -> None:
    """TODO(你)：断言 3 个以上连续换行被压成 2 个。

    "a\\n\\n\\n\\n\\nb"  →  "a\\n\\nb"
    """
    raw = "a\n\n\n\n\nb"
    result = normalize_whitespace(raw)
    assert result == "a\n\nb"


def test_normalize_is_idempotent() -> None:
    """TODO(你)：``normalize(normalize(x)) == normalize(x)``。

    幂等是这类清洗函数最重要的性质：跑两遍和跑一遍结果一样，
    你才敢在流水线的多个环节放心重复调用它。
    拿一段脏文本测（含 \\r\\n、多余空行、行尾空格、零宽字符）。
    """
    dirty_text = "你\u200b好\r\n这行末尾有空格   \n\n\n\n下一段\ufeff"
    once = normalize_whitespace(dirty_text)
    twice = normalize_whitespace(once)
    assert once == twice


def test_clean_document_preserves_doc_id() -> None:
    """TODO(你)：断言清洗后 doc_id 不变、source 不变、text 已被清洗。"""
    original_doc = Document(doc_id="doc-007", text="a\r\nb\u200bc   ", source="test-source")
    cleaned_doc = clean_document(original_doc)
    assert cleaned_doc.doc_id == original_doc.doc_id
    assert cleaned_doc.source == original_doc.source
    # 别只断言「变了」——那连「清洗成垃圾」都能通过。
    # 直接钉死清洗后的确切内容，强度高一个数量级。
    assert cleaned_doc.text == "a\nbc"


def test_clean_document_records_original_char_count() -> None:
    """TODO(你)：断言 metadata["original_char_count"] == 原始文本的长度。"""
    raw_text = "abc\r\n123\u200b"
    original_doc = make_document(text=raw_text)
    cleaned_doc = clean_document(original_doc)
    assert cleaned_doc.metadata["original_char_count"] == len(raw_text)


def test_clean_document_does_not_mutate_original() -> None:
    """TODO(你)：清洗完之后，**原 Document 的 text 必须原封不动**。

    Document 不是 frozen 的，所以「顺手改一下原对象」很容易发生。
    这条测试钉住「返回新对象、不动旧的」这条约定。
    """
    original_text = "a\r\nb\u200b"
    doc = make_document(text=original_text)
    _ = clean_document(doc)
    # 原对象不能被修改
    assert doc.text == original_text


def test_clean_document_rejects_blank_result() -> None:
    """TODO(你)：传一个全是空白 + 零宽字符的文本，断言抛 ValueError。

    这验证的是「重新构造会重新校验」：
    如果实现图省事写成 ``document.text = cleaned``，pydantic 不会重新校验，
    你会拿到一个 text 为空字符串的非法 Document，而且毫不知情。
    """
    blank_raw = "\u200b  \ufeff \r\n\t"
    doc = make_document(text=blank_raw)
    with pytest.raises(ValueError):
        clean_document(doc)
