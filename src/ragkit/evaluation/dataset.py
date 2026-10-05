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
    """一条评估样本：一个问题 + 它认为相关的结果 ID。

    ``relevant_ids`` 默认按 ``chunk_id`` 比对，也可以在 :func:`evaluate` 里
    用 ``match_on="doc_id"`` 改成按文档比对（只有文档级标注时更省事）。

    去重时**保留原顺序**：直接返回 set 的话，dump 回 JSONL 时顺序每次都变，
    文件 diff 会一直抖，测试也不稳定。
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
    """从 JSONL 文件读评估集。"""
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
    """把评估集写成 JSONL，返回写入条数。"""
    p = Path(path)
    lines = [sample.model_dump_json() for sample in samples]
    # 这里是**一个反斜杠**："\n" 才是换行符。
    # 写成 "\\n" 的话，join 出来的是字面量反斜杠 + n，整个文件会变成一行。
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(samples)
