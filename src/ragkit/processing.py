"""文本清洗：把解析出来的原始文本整理成「适合切分」的样子。

⚠️ 这个模块从头到尾是**同步**的，而且**必须**同步。

   清洗是纯字符串运算：不读文件、不调网络，没有任何「等待」发生。
   给纯 CPU 函数套 async 只会增加调用方的负担，一点并发收益都没有。
   （想在大量文档上并行清洗，正确做法是**多进程**，不是 async —— M3 结尾细说。）

   对照 parsing 包：Parser 要读磁盘（IO，会等待）→ async；
   Processor 只算字符串（CPU，不等待）→ sync。
   **判断标准是「会不会等待」，不是「活儿重不重」。**
"""

from __future__ import annotations

import re

from .schemas import Document

__all__ = ["normalize_whitespace", "clean_document"]

# 常见的零宽字符：零宽空格、零宽非连接符、零宽连接符、BOM。
# 它们肉眼看不见，却会让 "你好" in text 这种判断失败，
# 也会让 stable_id 对「看起来一样」的内容算出不同的 id。
_ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"


def normalize_whitespace(text: str) -> str:
    """把文本里的空白整理成规范形式。"""

    # Step1: 统一换行，顺序不能颠倒！先 \r\n，再单独 \r
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step2: 删除所有零宽字符
    for char in _ZERO_WIDTH:
        t = t.replace(char, "")

    # Step3: 每行右侧空白删除
    t = "\n".join(line.rstrip() for line in t.split("\n"))

    # Step4: 3个及以上连续换行压缩为两个换行
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Step5: 整体首尾strip
    t = t.strip()

    return t


def clean_document(document: Document) -> Document:
    """清洗一个 Document，返回**新的** Document，不改原对象。"""
    cleaned = normalize_whitespace(document.text)
    return Document(
        doc_id=document.doc_id,
        text=cleaned,
        source=document.source,
        metadata={**document.metadata, "original_char_count": len(document.text)},
    )
    # raise NotImplementedError("TODO: 返回新的 Document，见上面的形状")
