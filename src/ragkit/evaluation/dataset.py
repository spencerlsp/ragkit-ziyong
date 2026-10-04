"""评估数据集：问题 + 标准答案 + 应当被检索到的目标。

用 JSONL 存（M3 的 export 也是这个格式）。一行一条：

    {"question": "公司的年假政策是什么？", "relevant_ids": ["a1b2c3", "d4e5f6"]}

为什么不用 CSV：这条记录里有数组字段，CSV 表达起来要么加转义要么拆表，
而 JSONL 天然就是「一行一个 JSON 对象」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..errors import EvaluationError

__all__ = ["EvalSample", "load_dataset", "dump_dataset"]


class EvalSample(BaseModel):
    """一条评估样本。

    ★ 照着 Document / Chunk 的套路写（M1 的模仿案例）★

    字段：
        question: str
            —— 用来检索的问题。
        relevant_ids: list[str]
            —— 正确结果的 ID 列表。默认按 ``chunk_id`` 比对，
               也可以在 evaluate() 里指定按 ``doc_id`` 比对
               （很多人只有文档级标注，标 chunk 太费劲）。
        metadata: dict[str, Any] = Field(default_factory=dict)
            —— 来源、难度、类别…… 以后想按类别分组看指标时会用到。

    校验器：
        @field_validator("question")
        @classmethod
        def _question_must_not_be_blank(cls, value: str) -> str:
            —— strip 之后不能为空，和 Document.text 一样（照抄 M1 的写法）。

        @field_validator("relevant_ids")
        @classmethod
        def _dedupe_ids(cls, value: list[str]) -> list[str]:
            —— **去重但保持原顺序**：``list(dict.fromkeys(value))``。
               dict 从 3.7 起保序，这是「保序去重」最简洁的写法。
               为什么保序而不是直接 set：如果直接返回 set，
               dump 回 JSONL 时顺序每次都不一样，
               文件 diff 会一直变（而且测试也会不稳定）。
    """

    question: str
    relevant_ids: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question 不能为空")
        return question

    @field_validator("relevant_ids")
    @classmethod
    def _dedupe_ids(cls, value: list[str]) -> list[str]:
        # 去重，保留第一次出现的顺序
        return list(dict.fromkeys(value))


def load_dataset(path: str | Path) -> list[EvalSample]:
    """从 JSONL 文件读评估集。

    TODO(你)：四步。

        1) ``p = Path(path)``，文件不存在就抛 EvaluationError。

        2) 读文本：``text = p.read_text(encoding="utf-8")``

           ★ 注意这里是**同步**读，和 M2 的 read_text_async 不一样。
             为什么这次可以同步：评估集通常是几百行的小文件，
             而且它是**一次性的准备动作**，不在热路径上。
             给这种调用套异步，只会让调用方多写一个 await，
             换不来任何并发收益。
             **「会不会阻塞」是判断异步的唯一标准，但这个标准要考虑量级** ——
             几毫秒的同步读，不值得为它把整个调用链染成 async。

        3) 逐行解析。注意三个细节：
             * 用 ``text.splitlines()`` 而不是 ``.split("\\n")`` ——
               splitlines 能同时处理 \\r\\n / \\r / \\n 三种换行。
             * **跳过空行**（``if not line.strip(): continue``）——
               文件末尾常有一个空行，JSONL 的容忍度是它的优点之一。
             * 解析失败要带上**行号**：
                   raise EvaluationError(f"第 {lineno} 行不是合法 JSON: {exc}", source=str(p))
               没有行号的报错，用户面对一个 500 行的文件只能一行行数。

        4) ``json.loads(line)`` 之后用 ``EvalSample.model_validate(obj)`` 构造。
           校验失败同样要带上行号 —— pydantic 的报错很详细但不知道是哪一行。

           返回 ``list[EvalSample]``。
    """
    # 1. 检查文件是否存在
    p = Path(path)
    if not p.is_file():
        raise EvaluationError(f"评估文件不存在：{path}")

    # 2. 同步读取全部文本
    text = p.read_text(encoding="utf-8")

    samples: list[EvalSample] = []
    # 3. 逐行处理，enumerate 从1开始计数行号
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        try:
            obj = json.loads(stripped_line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"第 {lineno} 行不是合法 JSON: {exc}", source=str(p)) from exc

        # 4. 用Pydantic校验并构造EvalSample
        try:
            sample = EvalSample.model_validate(obj)
        except Exception as exc:
            raise EvaluationError(f"第 {lineno} 行数据校验失败: {exc}", source=str(p)) from exc

        samples.append(sample)

    return samples


def dump_dataset(samples: list[EvalSample], path: str | Path) -> int:
    """把评估集写成 JSONL，返回写入条数。

    TODO(你)：三行。

        p = Path(path)
        lines = [sample.model_dump_json() for sample in samples]
        <用换行符把 lines 连起来，末尾再补一个换行符，写进文件>
        return len(samples)

    ⚠️ 中间那一步**故意不写成可抄的代码示例**：
        docstring 里要显示出「反斜杠 + n」这两个字符，源码里就得写成双反斜杠；
       而照着源码抄的人会多抄一个，换行符于是变成字面量的反斜杠加 n，
       整个文件挤成一行。这个坑在本项目已经踩过四次（\\x1f、\\n\\n、\\u200b、\\n），
       所以从这条开始，涉及转义的示例一律只描述、不给可抄的写法。
       正确写法见函数体里的注释。

    两个细节：
      * 末尾要补一个换行。不补的话，用 ``cat`` 或 ``Get-Content`` 看时
        最后一行会和 shell 提示符粘在一起；而且很多「按行处理」的工具
        对没有结尾换行的文件处理不一致。
      * ``model_dump_json()`` 一行搞定序列化，不用手写 json.dumps。

    为什么要有这个函数：评估集你是要**手写和维护**的。
    能把它读进来、改一改、再写回去，比手工拼 JSON 靠谱得多。
    """
    p = Path(path)
    lines = [sample.model_dump_json() for sample in samples]
    # 这里是**一个反斜杠**："\n" 才是换行符。
    # 写成 "\\n" 的话，join 出来的是字面量反斜杠 + n，整个文件会变成一行。
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(samples)
