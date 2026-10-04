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
    """把文本里的空白整理成规范形式。

    TODO(你)：按顺序做五步。

        1) 统一换行符：把 "\\r\\n" 和 "\\r" 都换成 "\\n"。
           —— Windows 的 txt 是 \\r\\n，老 Mac 是 \\r，PDF 抽出来是 \\n。
              不统一的话，后面按 "\\n\\n" 找段落边界永远找不到。
           提示：``text.replace("\\r\\n", "\\n").replace("\\r", "\\n")``
           ⚠️ 顺序不能反！先换 \\r\\n 再换单个 \\r，否则 \\r\\n 会被拆成两个 \\n。

        2) 去掉零宽字符：遍历 _ZERO_WIDTH 里的每个字符，逐个 replace 成 ""。

        3) 逐行去掉行尾空白：
               "\\n".join(line.rstrip() for line in text.split("\\n"))
           —— 行尾空格是 PDF 抽取的常见垃圾，会让 token 数虚高。

        4) 把 3 个及以上连续换行压成 2 个：``re.sub(r"\\n{3,}", "\\n\\n", text)``
           —— 解析出来的空行数量全凭运气，把段落间隔固定成「一个空行」
              （也就是 "\\n\\n"），M3 的切分器才能稳定地按段落切。

        5) 去掉首尾空白：``.strip()``

        返回整理后的文本。

    这个函数最重要的性质是**幂等**：``f(f(x)) == f(x)``。
    幂等的函数你才敢在流水线里重复调用它。
    """

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
    """清洗一个 Document，返回**新的** Document，不改原对象。

    TODO(你)：按下面的形状实现，然后想清楚注释里的三个为什么。

        cleaned = normalize_whitespace(document.text)
        return Document(
            doc_id=document.doc_id,          # ← 注意这里没有重算
            text=cleaned,
            source=document.source,
            metadata={**document.metadata, "original_char_count": len(document.text)},
        )

    三个点必须想明白：

      1) **为什么保留原 doc_id，而不是重新算**
         doc_id 表示「这份数据来自哪个文件的哪一次解析」，是**身份**；
         清洗只是对内容做了一次变形，不该改变身份。
         而且 Chunk.doc_id 要能指回这个 Document，身份一飘就对不上了。
         （真要重新算：改一次清洗规则，Milvus 里的老数据就全成了孤儿。）

      2) **为什么重新构造，而不是 document.text = cleaned**
         pydantic 默认**赋值时不重新校验**（除非打开 validate_assignment）。
         直接改属性会绕过 text 的「非空」校验器 ——
         万一清洗把内容清成了空字符串，你会得到一个不合法的 Document 而毫不知情。
         重新构造会老老实实跑一遍校验，改坏了当场报错。

      3) **为什么记 original_char_count**
         「清洗掉了多少内容」是个很有用的诊断信号。
         如果一份 PDF 原始 3 万字、清洗后只剩 200 字，
         大概率是解析出了问题，这时候该去看一眼，而不是继续往下走。
    """
    cleaned = normalize_whitespace(document.text)
    return Document(
        doc_id=document.doc_id,
        text=cleaned,
        source=document.source,
        metadata={**document.metadata, "original_char_count": len(document.text)},
    )
    # raise NotImplementedError("TODO: 返回新的 Document，见上面的形状")
