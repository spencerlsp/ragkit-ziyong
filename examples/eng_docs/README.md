# eng_docs —— 工程文档语料

用真实工程文档替换了原先那份《劳动法》，用来做**有干扰项**的检索与评估。

> ⚠️ **语料正文不随仓库分发。** 下面这 25 篇 `.md` 是第三方文档（MIT），
> 体积也不小，所以仓库里**只带这份说明和 `eval_langchain_*.jsonl`**（标注和
> 评估集是我们自己的内容）。要把实验跑起来，先按下面的「获取方式」把文件下载到
> `examples/eng_docs/langchain/`，再照「怎么用」执行 —— 评估集里的 `chunk_id`
> 是切分结果的哈希，**必须是同一份语料 + 同一套切分参数**才对得上。

## langchain/ —— LangChain 官方工程文档

来源是 LangChain 官方仓库里的 `openwiki/` 目录（面向框架开发者与维护者的工程文档：
架构分层、中间件、工具系统、消息与内容块、流式输出、结构化输出、回调、
Agent 执行流程、测试与 CI 等）。

| 项目 | 说明 |
| --- | --- |
| 仓库 | <https://github.com/langchain-ai/langchain> |
| 路径 | `openwiki/` |
| commit | `b060a7466482ed5dac5fa1ef12e3ad21cccce385`（master，2026-10-09） |
| 文件数 | 25 篇 `.md` |
| 规模 | 571245 字符 → 默认切分参数下 **848 个 chunk**（min 234 / avg 786 / max 918） |
| 许可 | MIT（随仓库 LICENSE） |
| 获取方式 | 按上述 commit 从 `raw.githubusercontent.com` 逐文件下载，**内容未改动** |

两点说明：

1. 每篇文件开头的 YAML frontmatter（含 `sources:` 的 `repo://` 映射）**保留原样**，
   只占全文约 6%。真实文档本来就有这层元数据，留着更接近实际检索场景。
2. `index.md` 里逐条列出了所有文档的摘要，而每个文件的 frontmatter 里又有自己的
   `description`，同一段文字在语料里出现多次——这是**天然的近似重复干扰项**，
   正好用来观察检索会不会被摘要层吸走注意力。

## 怎么用

`parse` / `ingest` 直接收目录：会递归子目录、只挑解析器认得的后缀、结果排序。
所以命令跟 shell 无关，bash / PowerShell / cmd 里写法都一样。

```bash
# 离线：只看解析与切分，不需要 Milvus 和 API key
uv run ragkit parse examples/eng_docs/langchain
uv run ragkit split examples/eng_docs/langchain/architecture.md --out chunks.jsonl

# 入库整个语料（848 个 chunk）
uv run ragkit ingest examples/eng_docs/langchain --reset

# 检索验证
uv run ragkit query "how does middleware wrap model calls" -k 3
uv run ragkit query "how does middleware wrap model calls" -k 3 --mode hybrid --rerank
```

语言是英文，`BAAI/bge-m3` 这类多语模型可以直接用；问题与文档用语对齐会更贴近真实场景。

## 评估集：主文件 + 按切分映射

`eval_langchain_master.jsonl` 是**出题的地方**，31 条：

| 类型 | 条数 | 说明 |
| --- | --- | --- |
| `fact` | 21 | 单点事实题，问法偏口语，不照抄文档用词 |
| `multi_hop` | 5 | 需要跨 2 篇文档综合，例如「接入新 provider 要动哪些地方」 |
| `refusal` | 5 | 语料里没有答案（定价、star 数、性能基准、云成本），用来查会不会硬编 |

主文件只记 `question` + `evidence`（答案原文片段），**不写 `chunk_id`** ——
因为 `chunk_id` 是切分结果的哈希，换切分器就全变了。
入库前用脚本按当前切分结果映射一次：

```bash
# 专属切分器
uv run python scripts/build_eval_variants.py --out examples/eng_docs/eval_langchain_section.jsonl

# 基线切分器（对照组）
uv run python scripts/build_eval_variants.py --splitter recursive --chunk-overlap 120 \
    --out examples/eng_docs/eval_langchain_recursive.jsonl
```

标注规则是「**凡是包含答案原文的块都算相关**」。脚本会顺手做两件检查：
依据有没有命中（没命中直接报错退出），以及单条依据命中的块是不是太多
（超过 `--max-hits` 就告警 —— 依据太泛会让指标虚高）。

生成的 `<out>` 可以直接喂给 `ragkit eval`；拒答题单独写在
`<out>.refusals.jsonl`，**不要混进检索指标** ——
它们的 `relevant_ids` 是空的，混进去只会让 hit_rate 全归零，
而不是量出「该不该拒答」。

⚠️ 出题时踩过的一个坑：**别拿标题当 `evidence`**。
专属切分器会把标题路径补进该节下每一块的前缀里，
所以「Content Blocks: Unified Multimodal Representation」这种标题
会命中整节的十几块，标注瞬间失去区分度。用正文原句，别用标题。
