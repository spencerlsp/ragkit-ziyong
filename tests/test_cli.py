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
from types import SimpleNamespace

import pytest

from ragkit.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, _build_retriever, build_parser, main
from ragkit.errors import IndexingError
from ragkit.pipeline import IngestResult
from ragkit.splitting import MarkdownSectionSplitter


def write(tmp_path: Path, name: str, text: str) -> Path:
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


def test_parser_exposes_all_commands() -> None:
    """十个命令一个都不能少 —— 少一个用户就得多写一段脚本。"""
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set(next(iter(actions)).choices)  # type: ignore[arg-type]
    assert commands == {
        "parse",
        "split",
        "ingest",
        "query",
        "count",
        "info",
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
    """``overlap >= chunk_size`` 是**参数写错了**（退出码 2），不是业务失败（1）。

    切分器抛的是 ``SplitError``，漏到 main 会被当成业务失败 ——
    CLI 在边界上把它重新分类成 ``UsageError``。
    """
    target = write(tmp_path, "a.md", "一些内容")

    code = main(["split", str(target), "--chunk-size", "50", "--chunk-overlap", "80"])

    assert code == EXIT_USAGE
    assert "用法错误" in capsys.readouterr().err


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


def test_split_section_splitter_prefixes_heading_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--splitter section`` 要按标题分节，并给每块补 ``[doc | section]`` 前缀。

    这是它和 recursive 的核心差别：正文里多出一行标题路径，
    检索时标题词才参与匹配。这里加上 ``--no-merge`` 是为了让 Alpha / Beta
    各成一块 —— 小兄弟节在默认设置下会被合并，就看不到单节前缀了。
    """
    target = write(
        tmp_path,
        "a.md",
        "# Title\n\n## Alpha\n\nAlpha body text.\n\n## Beta\n\nBeta body text.\n",
    )
    export = tmp_path / "chunks.jsonl"

    code = main(
        [
            "split",
            str(target),
            "--splitter",
            "section",
            "--no-merge",
            "--chunk-overlap",
            "0",
            "--out",
            str(export),
        ]
    )

    assert code == EXIT_OK
    rows = [json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()]
    assert any(row["text"].startswith("[doc: Title | section: Alpha]") for row in rows)
    assert "块" in capsys.readouterr().out


def test_split_section_splitter_can_drop_context(tmp_path: Path) -> None:
    """``--no-context`` 是消融开关：除了不补前缀，其余行为不变。"""
    target = write(tmp_path, "a.md", "# Title\n\n## Alpha\n\nAlpha body text.\n")
    export = tmp_path / "chunks.jsonl"

    code = main(
        [
            "split",
            str(target),
            "--splitter",
            "section",
            "--no-context",
            "--out",
            str(export),
        ]
    )

    assert code == EXIT_OK
    rows = [json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(not row["text"].startswith("[doc:") for row in rows)


def test_split_honours_zero_overlap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--chunk-overlap 0`` 必须真的按 0 走。

    以前这里写的是 ``args.chunk_overlap or settings.chunk_overlap``：
    0 是 falsy，会被静默换成配置里的 120 —— 参数明明传了却不生效，
    切分器也不报错，属于最难自查的那类问题。
    而 section 切分器推荐的正是 overlap=0，必须保证它真的落地。
    """
    target = write(tmp_path, "a.md", "内容。" * 200)

    assert main(["split", str(target), "--chunk-overlap", "0", "--json"]) == EXIT_OK

    payload = json.loads(capsys.readouterr().out)
    assert payload["chunk_overlap"] == 0
    assert payload["splitter"] == "recursive"


def test_parser_exposes_splitter_and_section_aware_flags() -> None:
    """两组新开关要能被解析出来 —— CLI 的价值就在这些开关上。"""
    args = build_parser().parse_args(
        ["ingest", "a.md", "--splitter", "section", "--no-context", "--no-merge"]
    )
    assert args.splitter == "section"
    assert args.no_context is True
    assert args.no_merge is True

    args = build_parser().parse_args(
        ["eval", "eval.jsonl", "--section-aware", "--candidate-factor", "6"]
    )
    assert args.section_aware is True
    assert args.candidate_factor == 6
    # 没传就是 None —— 真正的默认值在 _build_retriever 里落
    # （故意不写进 argparse.default，这样才分得清「用户显式传了 0.5」
    #  和「用户没传」，否则「传了参数却没开开关」就发不出提示）。
    assert args.section_boost is None
    assert args.max_per_section is None

    assert build_parser().parse_args(["query", "问题"]).section_aware is False


def test_parse_accepts_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """帮助里写着「文件或目录」，那就得真的支持目录。

    只认文件的话，用户被迫在 shell 里拼 glob，
    而 PowerShell 和 bash 的展开语法完全不同 —— 换个终端就翻车。
    这里顺带确认：递归子目录、且只收解析器认得的后缀。
    """
    (tmp_path / "a.md").write_text("内容 A。", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("内容 B。", encoding="utf-8")
    (tmp_path / "skip.xyz").write_text("不支持的后缀", encoding="utf-8")

    assert main(["parse", str(tmp_path)]) == EXIT_OK

    out = capsys.readouterr().out
    assert "2 个文件" in out
    assert "skip.xyz" not in out


def test_parse_rejects_directory_without_supported_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """目录里一个能解析的文件都没有时要报错，而不是静默地「0 个文件」。"""
    (tmp_path / "only.xyz").write_text("x", encoding="utf-8")

    assert main(["parse", str(tmp_path)]) == EXIT_ERROR
    assert "没有可解析的文件" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 参数分类：用法错误(2) 还是业务失败(1)
# ---------------------------------------------------------------------------


def test_section_splitter_defaults_to_zero_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--splitter section`` 的默认 overlap 必须是 0，不能沿用 .env 的 120。

    这不是口味问题：``scripts/build_eval_variants.py`` 生成 section 版评估集时
    用的就是 0，而评估集里的 ``chunk_id`` 是切分结果的哈希。两处默认值一旦
    不一致，用户按文档敲一遍、再自己敲一遍，得到的就是两套对不上的标注 ——
    指标悄悄低一截，一句报错都没有。
    """
    target = write(tmp_path, "a.md", "# 标题\n\n正文内容。\n")

    assert main(["split", str(target), "--splitter", "section", "--json"]) == EXIT_OK

    payload = json.loads(capsys.readouterr().out)
    assert payload["splitter"] == "section"
    assert payload["chunk_overlap"] == 0


def test_chunk_size_below_config_overlap_says_where_it_came_from(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--chunk-size 80`` 配 .env 的 overlap=120 时，要说清 120 是从哪冒出来的。

    以前这里只有切分器那句「chunk_overlap(120) 必须小于 chunk_size(80)」：
    数字对不上，但用户根本不知道 120 是谁给的，只能盯着命令行发呆。
    （而且以前那条「建议加 --chunk-overlap」的提示压根打印不出来 ——
      它在切分器构造之后，而构造时就已经抛异常了。）
    """
    target = write(tmp_path, "a.md", "内容。")

    code = main(["split", str(target), "--chunk-size", "80"])

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "CHUNK_OVERLAP" in err
    assert "--chunk-overlap" in err


def test_empty_query_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """空查询是参数没给对（2），不是检索失败（1）。"""
    assert main(["query", "   "]) == EXIT_USAGE
    assert "用法错误" in capsys.readouterr().err


def test_bad_section_boost_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """``--section-boost -1`` 的校验在检索器构造函数里，CLI 负责重新分类。"""
    assert main(["query", "问题", "--section-aware", "--section-boost", "-1"]) == EXIT_USAGE
    assert "用法错误" in capsys.readouterr().err


def test_section_only_flags_warn_under_recursive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--no-context`` 配 recursive 会被忽略 —— 必须说出来，不能静默。"""
    target = write(tmp_path, "a.md", "# 标题\n\n正文。\n")

    assert main(["split", str(target), "--no-context"]) == EXIT_OK
    assert "--no-context" in capsys.readouterr().err


def test_section_aware_knobs_warn_when_the_switch_is_off(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """调了 ``--max-per-section`` 却没开 ``--section-aware``：参数会静默失效。

    这里直接调 ``_build_retriever`` 并塞一个假门面 —— 只能拿到 ``retriever``
    一个属性，所以不需要 Milvus，也不需要真的造一个 Ragkit。
    """
    args = build_parser().parse_args(["query", "问题", "--max-per-section", "3"])
    sentinel = object()

    assert _build_retriever(args, SimpleNamespace(retriever=sentinel)) is sentinel
    assert "--max-per-section" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 不碰数据库的命令：用一个假门面替掉 Ragkit
# ---------------------------------------------------------------------------


class FakeRagkit:
    """不连 Milvus 的 ``Ragkit`` 替身，只实现 CLI 用到的那几个方法。

    能做到这么短，是因为 CLI 和门面之间的接口很窄：``async with`` +
    ``ingest`` / ``count``。为了验证「CLI 打出来的 JSON 对不对」
    去起一个 Milvus，成本远大于收益。
    """

    def __init__(self, *, count: int = 0, error: Exception | None = None, **kwargs: object) -> None:
        self.kwargs = kwargs  # 记下 CLI 传进来的 splitter，断言用
        self._count = count
        self._error = error

    async def __aenter__(self) -> FakeRagkit:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def ingest(
        self, paths: object, *, reset: bool = False, flush: bool = True
    ) -> IngestResult:
        if self._error is not None:
            raise self._error
        splitter = self.kwargs.get("splitter")
        return IngestResult(
            files=2,
            chunks=3,
            written=3,
            doc_ids=["d1", "d2"],
            # 直接读真实切分器对象的属性：这样断言才等于在验证
            # 「CLI 真的把它传进来了」，而不是验证我在测试里写的常数。
            splitter=getattr(splitter, "name", "?"),
            chunk_size=int(getattr(splitter, "chunk_size", 0)),
            chunk_overlap=int(getattr(splitter, "chunk_overlap", 0)),
        )

    async def count(self) -> int:
        if self._error is not None:
            raise self._error
        return self._count


def test_ingest_json_records_the_chunking_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ingest --json`` 要带上切分参数 —— 它是排查「指标全 0」的唯一证据。

    人肉跑的时候会有一行提示告诉你「这批是按 section 切的」；
    但 CI 用的是 ``--json``，而 ``IngestResult`` 以前只有
    files/chunks/written/doc_ids 四个字段 —— **机器路径反而什么都没记**。
    """
    target = write(tmp_path, "a.md", "# 标题\n\n正文。\n")
    made: list[FakeRagkit] = []

    def factory(**kwargs: object) -> FakeRagkit:
        rag = FakeRagkit(**kwargs)
        made.append(rag)
        return rag

    monkeypatch.setattr("ragkit.cli.Ragkit", factory)

    assert main(["ingest", str(target), "--splitter", "section", "--json"]) == EXIT_OK

    payload = json.loads(capsys.readouterr().out)
    assert payload["splitter"] == "section"
    assert payload["chunk_size"] == 800
    assert payload["chunk_overlap"] == 0
    assert isinstance(made[0].kwargs["splitter"], MarkdownSectionSplitter)


def test_info_prints_config_and_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``info`` 要一次打全「当前配置 + 库里条数」，这是排查的第一步。"""
    monkeypatch.setattr("ragkit.cli.Ragkit", lambda **kwargs: FakeRagkit(count=848))

    assert main(["info"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "Collection" in out
    assert "数据库" in out
    assert "记录数：848" in out

    assert main(["info", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 848
    assert payload["error"] is None
    # JSON 给的是**完整**配置（字段名与 .env 里的键一一对应），
    # 这样「两个项目的配置一致吗」可以直接 diff
    settings = payload["settings"]
    assert settings["milvus_collection"]
    assert settings["milvus_uri"].startswith("http")
    assert settings["top_k"] > 0
    # 密钥只以「已配置 / 空」出现 —— 这个输出是会被贴出去的
    assert settings["openai_embedding_api_key"] in ("已配置", "")
    assert settings["rerank_api_key"] in ("已配置", "")


def test_info_still_prints_config_when_milvus_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """连不上 Milvus 时配置照打 —— 那正是最需要看「我连的是哪个库」的时候。"""
    monkeypatch.setattr(
        "ragkit.cli.Ragkit",
        lambda **kwargs: FakeRagkit(error=IndexingError("连不上", source="ragkit_chunks")),
    )

    assert main(["info"]) == EXIT_ERROR

    out = capsys.readouterr().out
    assert "Collection" in out
    assert "取不到" in out
