"""CLI 的测试。

只测**不碰数据库**的命令（parse / split / 参数解析 / 退出码）——
碰数据库的那几条在真 Milvus 上测才有意义，属于集成测试的范畴。

直接调 ``main([...])`` 而不是起子进程：快得多，而且退出码就是返回值，
不用去解析 shell 的退出状态。只有「argparse 自己抛 SystemExit」的情况
才需要 ``pytest.raises(SystemExit)``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.cli import EXIT_ERROR, EXIT_OK, build_parser, main


def write(tmp_path: Path, name: str, text: str) -> Path:
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


def test_parser_exposes_all_commands() -> None:
    """九个命令一个都不能少 —— 少一个用户就得多写一段脚本。"""
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set(next(iter(actions)).choices)  # type: ignore[arg-type]
    assert commands == {
        "parse",
        "split",
        "ingest",
        "query",
        "count",
        "delete",
        "drop",
        "eval",
        "mcp",
    }


def test_parse_reports_char_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = write(tmp_path, "a.md", "你好，世界。这是一份测试文档。")

    assert main(["parse", str(target)]) == EXIT_OK

    out = capsys.readouterr().out
    assert "1 个文件" in out
    assert str(target) in out


def test_parse_json_output_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = write(tmp_path, "a.md", "内容")

    assert main(["parse", str(target), "--json"]) == EXIT_OK

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload[0]["chars"] == 2
    assert payload[0]["source"] == str(target)


def test_split_writes_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``split --out`` 是调切分参数时最常用的动作，必须完全离线可用。"""
    target = write(tmp_path, "a.md", "第一段内容。" * 40)
    export = tmp_path / "chunks.jsonl"

    code = main(
        [
            "split",
            str(target),
            "--chunk-size",
            "80",
            "--chunk-overlap",
            "20",
            "--out",
            str(export),
        ]
    )

    assert code == EXIT_OK
    lines = export.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 1
    # 每行都是合法 JSON，且带 chunk_id
    assert all(json.loads(line)["chunk_id"] for line in lines)
    assert "块" in capsys.readouterr().out


def test_split_rejects_bad_params(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """overlap >= chunk_size 会抛 SplitError，CLI 要翻译成退出码 1 而不是 traceback。"""
    target = write(tmp_path, "a.md", "一些内容")

    code = main(["split", str(target), "--chunk-size", "50", "--chunk-overlap", "80"])

    assert code == EXIT_ERROR
    assert "错误" in capsys.readouterr().err


def test_missing_file_reports_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """文件不存在时不能甩一堆 ExceptionGroup traceback。

    parse 内部用 TaskGroup，异常会被打包成 ExceptionGroup；
    用户要看到的是一句人话 + 退出码 1。
    """
    code = main(["parse", str(tmp_path / "不存在.md")])

    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "错误" in err
    assert "TaskGroup" not in err


def test_unknown_command_is_a_usage_error() -> None:
    """写错命令是「用法错误」（退出码 2），不是「业务失败」（退出码 1）。"""
    with pytest.raises(SystemExit) as exc_info:
        main(["不存在的命令"])
    assert exc_info.value.code == 2
