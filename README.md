# ragkit

> 一个「能用、能懂、能改」的通用 RAG 工具包：**文件解析 → 清洗切分 → 向量化 → Milvus 索引 → 检索（稠密 / 混合 / 重排）→ 评估**。

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-206%20passed-brightgreen)
![Async](https://img.shields.io/badge/asyncio-first-orange)
![Vector%20DB](https://img.shields.io/badge/Milvus-2.5%2B-00A1EA)

---

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
  - [前置条件](#1-前置条件)
  - [安装](#2-安装)
  - [配置](#3-配置)
  - [第一段代码](#4-第一段代码)
- [使用指南](#使用指南)
  - [入库](#入库)
  - [查询](#查询)
  - [管理](#管理)
  - [离线开发](#离线开发不联网不花钱)
  - [不用门面：分层使用](#不用门面分层使用)
  - [评估](#评估)
  - [错误处理](#错误处理)
- [工作流程](#工作流程)
- [API 一览](#api-一览)
- [配置参考](#配置参考)
- [跑一遍完整示例](#跑一遍完整示例)
- [命令行（CLI）](#命令行cli)
  - [命令总览](#命令总览)
  - [通用约定](#通用约定)
  - [命令详解](#命令详解)
  - [切分与检索的开关](#切分与检索的开关)
  - [参数一致性](#参数一致性)
  - [四组对照实验](#四组对照实验)
  - [排查手册](#排查手册)
- [作为 MCP 服务端给 agent 用](#作为-mcp-服务端给-agent-用)
- [评估结果](#评估结果)
- [常见坑速查](#常见坑速查)
- [开发](#开发)
- [已知限制](#已知限制)
- [设计笔记](#设计笔记)

---

## 特性

| 能力 | 说明 |
|---|---|
| **多格式解析** | `.txt` / `.md` / `.pdf` / `.docx`，解析器可注册扩展 |
| **异步优先** | 解析 / 向量化 / 写入 / 检索全程异步，带限流、超时、指数退避重试 |
| **切分策略可选** | 递归切分（段落 → 换行 → 句号 → 字符）/ 按 Markdown 标题分节（工程文档）；保证 `"".join(pieces) == 原文` |
| **混合检索** | 稠密向量 + BM25 关键词，RRF 融合 —— 专治「PIP」「E-204」这类稀有关键词 |
| **重排（Rerank）** | 两阶段检索：召回 20~50 条候选 → cross-encoder 精排取 top-k |
| **可评估** | HitRate / Recall / MRR / NDCG，支持**按问题分组**看指标、跑批容错 |
| **可替换** | 解析器 / 切分器 / Embedder / 向量库 / 重排器全是 Protocol，换实现不改上层 |

---

## 快速开始

### 1. 前置条件

| 需要 | 版本 | 说明 |
|---|---|---|
| Python | 3.11+ | 代码里用了 `asyncio.timeout`、`TaskGroup`、`zip(strict=)` |
| [uv](https://docs.astral.sh/uv/) | 任意 | 包管理与虚拟环境 |
| Milvus | **2.5+** | BM25 / `hybrid_search` / `RRFRanker` 都是 2.5 才有的；本仓库在 **2.6.15** 上验证 |
| Embedding API | 任意 OpenAI 兼容端点 | 硅基流动 / DashScope / OpenAI / vLLM 都行 |
| Rerank API（可选） | 任意 `/rerank` 端点 | 只在用重排时才需要 |

Milvus 怎么起（Docker standalone / Zilliz Cloud 免费实例）见 [官方文档](https://milvus.io/docs)。
`MILVUS_URI` 默认指向 `http://localhost:19530`。

### 2. 安装

```bash
git clone <repo> && cd ragkit
uv sync
```

验证装好了：

```bash
uv run python -c "import ragkit; print(ragkit.__all__)"
uv run pytest -q          # 281 条测试，全部离线，不需要 Milvus 和 API key
```

### 3. 配置

复制 `.env.example` 为 `.env`，填上 key。**最小可用配置**只有四行：

```ini
OPENAI_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_EMBEDDING_API_KEY=sk-你的key
OPENAI_EMBEDDING_MODEL=BAAI/bge-m3
OPENAI_EMBEDDING_DIM=1024
```

完整配置（含重排、Milvus、切分参数）见 [配置参考](#配置参考)。

> ⚠️ **`OPENAI_EMBEDDING_DIM` 必须和模型的真实维度一致。**
> `BAAI/bge-m3` 是 1024，`text-embedding-3-small` 是 1536。
> 写错了不会立刻报错，而是会在**建 Milvus 表的时候**失败 ——
> 所以代码里设了三道校验（配置层 / 客户端层 / schema 层）把它拦在前面。

### 4. 第一段代码

```python
import asyncio
from ragkit import Ragkit


async def main() -> None:
    async with Ragkit() as rag:
        # ① 入库：解析 → 清洗 → 切分 → 向量化 → 写进 Milvus
        result = await rag.ingest(["docs/员工手册.md", "docs/报销制度.pdf"])
        print(f"入库 {result.files} 个文件 / {result.chunks} 个 chunk / {result.written} 条记录")

        # ② 查询：默认稠密检索
        hits = await rag.query("出差住宿费上限是多少", top_k=3)
        for rank, hit in enumerate(hits, start=1):
            print(f"{rank}. score={hit.score:.3f}  {hit.chunk.text[:60]}")


asyncio.run(main())
```

`Ragkit` 是**薄门面**：它只负责按顺序调用和管生命周期。五个组件都能注入替换
（`embedder=` / `indexer=` / `reranker=` / `splitter=`），测试和定制都靠它。
真正的东西都在各层里 —— 门面薄，是因为下层厚。

---

## 使用指南

### 入库

#### 基本用法

```python
result = await rag.ingest(["a.md", "b.pdf", "c.docx"])
```

`parse_files` 内部是**并发**的（`TaskGroup` + `Semaphore` 限流），所以传一百个文件也不用担心打爆线程池。

#### 返回值怎么读

`IngestResult` 给了四个字段，**四个都要看**：

```python
result.files  # 解析成功的文件数
result.chunks  # 切出来的 chunk 总数
result.written  # 真正写进 Milvus 的条数
result.doc_ids  # 这批数据的文档 ID（删除时要用，见「管理」）
```

| 现象 | 说明 |
|---|---|
| `files` 少于传入的文件数 | 不可能——有文件失败的话会直接抛异常，不会静默跳过 |
| `chunks` 远小于预期 | 切分粒度太大，或者文件本身很短 |
| `chunks` 多于 `written` | **写入阶段丢数据了**，要查 |

#### 幂等：同一份文件导入两次不会重复

入库的顺序固定是 **先按 `doc_id` 删除，再 upsert**：

```
delete(doc_id) → embed(texts) → upsert(rows) → flush
```

所以重复导入同一份文件会**覆盖**旧数据，而不是堆重复。反过来的顺序（先写后删）会把刚写的数据一起删掉。

#### `reset=True`：什么时候用

```python
await rag.ingest(["a.md"], reset=True)
```

它会先**删掉整张表**再重建。两种情况下必须用它：

1. **改了 schema**（比如给 collection 加了 BM25 稀疏字段）—— Milvus 建好的字段改不了；
2. 怀疑库里数据脏了，想推倒重来。

这正好说明了那条通用规律：**schema 是合同，合同改了就得推倒重来。**

#### 大规模导入：关掉 flush

```python
await rag.ingest(paths, flush=False)  # 一批一批写，最后统一刷
await rag._indexer.flush()  # ← 不推荐直接摸私有属性，见下方说明
```

每次小批量都 `flush` 会产出大量小 segment，反而拖慢查询。持续大批量导入时应该关掉它。

> 目前 `Ragkit` 没有暴露独立的 `flush()` 方法。要么保持默认（每批都刷，简单但慢），
> 要么直接用底层的 `MilvusIndexer`（见「不用门面：分层使用」）。

---

### 查询

#### 三种模式

```python
await rag.query("问题")  # 稠密（默认）
await rag.query("问题", mode="hybrid")  # 稠密 + BM25，RRF 融合
await rag.query("问题", mode="hybrid", rerank=True)  # 再加一层精排
```

**怎么选**：看你库里有没有「稀有关键词」和「同义改述」这两类查询。

| 你的查询长这样 | 建议 |
|---|---|
| 自然语言提问，用词和文档接近 | 稠密就够 |
| 带产品名 / 编号 / 缩写 / 错误码（`PIP`、`E-204`） | **开混合** —— BM25 补的就是这块 |
| 用户会用完全不同的说法问同一件事 | **开重排** —— cross-encoder 能看见语义与字面双重交集 |

实测数据见[评估结果](#评估结果)：混合把「稀有词」组从 0.667 拉到 1.000，重排把「改述」组从 0.908 拉到 1.000，**两者补的是不同的短板**。

#### 参数

```python
hits = await rag.query(
    "问题",
    top_k=5,  # 返回几条（默认取配置里的 TOP_K）
    mode="hybrid",  # "dense" | "hybrid"
    rerank=True,  # 是否精排（需要配置 RERANK_*）
    rerank_candidates=50,  # 精排前先召回多少条候选（默认 max(top_k*4, 20)）
    score_threshold=0.5,  # 过滤低分结果（见下方注意事项）
    filter_expr='doc_id == "abc"',  # 只在指定范围内检索
    total_timeout=30.0,  # 整次检索的截止时间（秒）
)
```

**`rerank_candidates` 为什么必须比 `top_k` 大**：

> 只把 top-3 拿去重排，它只能在 3 条里换顺序，**一点召回收益都拿不到**。
> 重排的全部价值来自「把本该进前列、却被向量检索挤出去的那几条捞回来」。

**`score_threshold` 的注意事项**（三种模式的分数**不是一个量纲**）：

| 模式 | 分数含义 | 量级 | 能不能设阈值 |
|---|---|---|---|
| 稠密 | 余弦相似度 | 0 ~ 1 | ✅ |
| 混合 | RRF 融合分 | 约 0 ~ 0.033 | ❌ **会直接报错** |
| 混合 + 重排 | 重排模型的相关度分 | 0 ~ 1 | ✅ |

混合模式传 `score_threshold` 时，ragkit 会**拒绝执行**而不是默默过滤：
拿 0.7 去过滤 RRF 分数会把结果全部滤光，而用户只会看到「搜不到任何东西」。

#### 只搜某几篇文档

```python
from ragkit import doc_filter

result = await rag.ingest(["手册.md", "制度.md"])
hits = await rag.query(
    "年假有几天",
    filter_expr=doc_filter(result.doc_ids[:1]),  # 只查第一篇
)
```

`doc_filter(["a", "b"])` 会生成 `doc_id in ["a", "b"]`。也可以手写任意 Milvus 过滤表达式。

> 过滤在**召回阶段**生效，所以它同时也提升了速度 —— 候选少了，算得也就快了。

---

### 管理

```python
await rag.count()  # 库里有多少条（近似值，用于健康检查）
await rag.delete_document(doc_id)  # 删掉某个文档的所有 chunk
await rag.drop()  # 删掉整张表
```

#### 删除单个文档

```python
result = await rag.ingest(["手册.md"])
await rag.delete_document(result.doc_ids[0])
```

> ⚠️ **`doc_id` 是「文件路径 + 文件内容」的哈希。**
> 文件内容一变，`doc_id` 就变了。所以「改完文件想删掉旧数据」**不能靠重新解析那个文件** ——
> 新算出来的 id 和当初入库的 id 对不上。
>
> 稳妥做法只有两个：
> 1. 保存 `ingest()` 返回的 `doc_ids`；
> 2. 直接整表重建：`await rag.drop()` + 重新 `ingest()`。
>
> （这也是为什么「评估集要和切分策略一起版本化」是同一条规律 ——
> 凡是内容哈希当 ID 的地方，改输入就等于换了一批身份。）

---

### 离线开发（不联网、不花钱）

开发阶段反复调切分参数时，每次都调 embedding 接口既慢又费钱。用本地假向量替代：

```python
from ragkit import Ragkit, split_document, clean_document, parse_file
from ragkit.embedding import HashingEmbedder


async def main() -> None:
    document = await parse_file("docs/手册.md")
    cleaned = clean_document(document)
    chunks = split_document(cleaned)

    async with Ragkit(embedder=HashingEmbedder(dim=1024)) as rag:
        await rag.ingest(["docs/手册.md"])
        hits = await rag.query("随便问问")
```

`HashingEmbedder` 把文本哈希成确定性的单位向量 —— **不联网、零依赖、完全可复现**。

> ⚠️ 它**没有语义**：把「猫」和「狗」丢进去，向量不会比「猫」和「火车」更接近。
> 所以它只能用来验证「管道通不通」，**不能用来评估检索质量**。

命令行版更省事：

```bash
uv run python scripts/ingest_demo.py examples/sample_doc.md --fake
```

---

### 不用门面：分层使用

`Ragkit` 只是把常见路径包起来。任何一层都可以单独用，也可以只替换其中一层。

#### 只做解析

```python
from ragkit import parse_file, parse_files

doc = await parse_file("a.pdf")  # -> Document(doc_id, text, source, metadata)
docs = await parse_files(["a.md", "b.pdf"])  # 并发解析
```

#### 只做切分，并肉眼检查结果

调切分参数时最有用的一步 —— **把切出来的 chunk 写到文件里看一眼**：

```python
from ragkit import parse_file, clean_document, split_document, write_chunks_jsonl
from ragkit.splitting import RecursiveSplitter

doc = clean_document(await parse_file("手册.md"))

splitter = RecursiveSplitter(chunk_size=350, chunk_overlap=50)
chunks = splitter.split(doc)
await write_chunks_jsonl(chunks, Path("chunks.jsonl"))
```

```bash
Get-Content chunks.jsonl | Select-Object -First 3     # 看前三条长什么样
```

#### 只做向量化

```python
from ragkit import create_embedder, embed_all

embedder = create_embedder()
try:
    vectors = await embed_all(embedder, ["第一段", "第二段"], batch_size=32)
finally:
    await embedder.aclose()
```

`embed_all` 负责分片、并发、保序、总超时 —— **返回顺序和输入严格一致**，这是后续所有事情的前提。

#### 自己管生命周期

```python
from ragkit import MilvusIndexer, create_embedder, create_reranker

embedder = create_embedder()
indexer = MilvusIndexer()
try:
    await indexer.ensure_collection()
    await indexer.upsert_chunks(chunks, vectors)
    await indexer.flush()
finally:
    await indexer.aclose()
    await embedder.aclose()
```

> 规矩很简单：**谁创建，谁销毁。** 注入给别人的组件绝不关 —— 那可能是别人还要用的资源。

#### 换实现

五个扩展点都是 Protocol，照着形状写个类就能替换：

| 扩展点 | 协议 | 内置实现 |
|---|---|---|
| 解析 | `parsing.Parser` | `TextParser` / `PdfParser` / `DocxParser` |
| 切分 | `splitting.Splitter` | `RecursiveSplitter` / `FixedSplitter` |
| 向量化 | `embedding.Embedder` | `OpenAICompatEmbedder` / `HashingEmbedder` |
| 向量库 | `indexing.VectorStore` | `MilvusIndexer` |
| 重排 | `retrieval.Reranker` | `HttpReranker` |

```python
from ragkit.parsing import register


class CsvParser:
    extensions = (".csv",)

    async def parse(self, path): ...


register(CsvParser())  # 之后 parse_file("x.csv") 就能用了，主流程一行没改
```

---

### 评估

评估回答的是「**我这次改动到底有没有用**」。没有它，调参只能靠感觉。

#### 1. 准备数据集

JSONL 格式，一行一条：

```jsonl
{"question": "远程办公每周最多几天？", "relevant_ids": ["a1b2c3d4e5f60718"]}
{"question": "报销每月几号截止？", "relevant_ids": ["9f8e7d6c5b4a3210"]}
```

`relevant_ids` 默认按 `chunk_id` 比对；只有文档级标注的话，用 `match_on="doc_id"`。

> 怎么拿到 `chunk_id`？先切分，然后 `chunks[0].chunk_id`。
> 或者用 `scripts/eval_demo.py` 的自动标注方式先跑通流程。

仓库里有一份标好的样板：[`examples/eval_sample.jsonl`](examples/eval_sample.jsonl)
（11 个问题，对应 `examples/sample_doc.md`，按**默认切分参数**标注）。

```bash
uv run ragkit ingest examples/sample_doc.md        # 用默认切分参数，别加 --chunk-size
uv run ragkit eval examples/eval_sample.jsonl -k 3
```

> ⚠️ **`relevant_ids` 里的 chunk_id 绑定在切分参数上。** 改了 `CHUNK_SIZE` / `CHUNK_OVERLAP`，
> chunk_id 全变，标注立刻失效 —— 指标会恒为 0，而且**不报任何错**。
> 这就是为什么真实项目里**评估集要和切分策略一起版本化**。
>
> 这份样板对应的是默认的 800/120，所以 2825 字的示例文档只切成 **4 块** ——
> 粒度太粗，top-3 几乎必然命中，指标没有区分度。它的定位是**格式示例 + 验证管道通不通**。
> 想看真实效果，把 `CHUNK_SIZE` 调小（比如 350）重新切分，再用
> `uv run python scripts/eval_demo.py examples/sample_doc.md --rerank` 重新标注并对比。

#### 2. 跑评估

```python
from ragkit import EvalSample, evaluate, load_dataset

samples = load_dataset("data/eval.jsonl")

async with Ragkit() as rag:
    dense = await evaluate(samples, rag.retriever, k=3, mode="dense")
    hybrid = await evaluate(samples, rag.retriever, k=3, mode="hybrid")

print(dense.summary())
print(hybrid.summary())
```

#### 3. 怎么读结果

```
[dense]  k=3 n=11  hit_rate=0.909  recall=0.909  mrr=0.864  ndcg=0.876
```

| 指标 | 回答的问题 | 什么时候看 |
|---|---|---|
| `hit_rate` | 前 k 条里**有没有**相关内容？ | 第一道体检。它很低，后面都不用看 |
| `recall` | 相关内容**捞全了没有**？ | 标准答案多于一个时 |
| `mrr` | 第一条相关内容排第几位？ | 只关心「最相关的能不能排第一」时 |
| `ndcg` | 排序质量如何？ | **最综合**，同时反映「捞到几个」和「排得怎么样」 |

> 📌 **指标全是 1.0 不是好消息，是「评估集太简单」。**
> 那说明所有方法都能拿满分，你什么结论都得不出来。
> 评估集的难度上限，决定了评估的价值上限。

#### 4. 一定要分组看

只有一个综合分，你永远得不出可执行的结论。把问题按类型分组之后规律才浮出来：

| 分组 | 稠密 | 混合 | 混合+重排 |
|---|---|---|---|
| 直问（和文档用同一批词） | 1.000 | 1.000 | 1.000 |
| 改述（换成同义词） | 0.908 | 0.908 | **1.000** |
| 稀有词（缩写 / 编号） | 0.667 | **1.000** | 1.000 |

综合分只差 0.09，看着「提升不大」；分组之后才知道**两块不同的短板被分别补上了**。

---

### 错误处理

所有异常都继承自 `RagkitError`，所以**只写一个 `except` 就能兜住全部**：

```python
from ragkit import RagkitError, ParseError

try:
    await rag.ingest(["a.md", "b.pdf"])
except ParseError as exc:  # 精确捕获某一种
    print(f"解析失败：{exc.source} — {exc}")
except RagkitError as exc:  # 兜住其余全部
    print(f"出错了：{exc}")
```

#### 异常类型与触发场景

| 异常 | 什么时候抛 |
|---|---|
| `ParseError` | 文件不存在 / 格式不支持 / PDF 是扫描件（抽不出文字） |
| `SplitError` | 切分参数非法（比如 `chunk_overlap >= chunk_size`） |
| `EmbeddingError` | embedding 接口失败 / 返回条数对不上 / 向量维度不匹配 |
| `IndexingError` | Milvus 建表 / 写入 / 删除 / flush 失败 |
| `RetrievalError` | 检索失败 / 查询向量维度不匹配 / 重排失败 / 检索超时 |
| `GenerationError` | LLM 生成失败（当前未接入） |
| `EvaluationError` | 评估集格式不对 / 评估集为空 |

每个异常都带 `source` 字段，指明「是谁出的错」（文件路径 / collection 名 / 模型名）：

```python
except RagkitError as exc:
    print(exc.source)   # "docs/a.md" 或 "ragkit_chunks" 或 "BAAI/bge-m3"
```

#### 一个容易踩的坑：`TaskGroup` 的 `ExceptionGroup`

批量解析用的是 `TaskGroup`，它会把子任务的异常打包成 `ExceptionGroup`（PEP 654），
这会导致普通的 `except ParseError` **抓不到**。`Ragkit` 在只有一个错误时会**自动解包**；
多个文件同时失败时保留 `ExceptionGroup`（那时你确实需要看到全部错误）：

```python
try:
    await rag.ingest(["a.md", "b.md", "c.md"])
except ParseError:
    ...  # 只有一个文件坏 → 直接捕获
except ExceptionGroup as exc:
    for error in exc.exceptions:
        print(error)
```

---

## 工作流程

```
  .md / .pdf / .docx
         │
         ▼  parse_files()           并发解析，TaskGroup + Semaphore 限流
     Document                       doc_id = stable_id(路径, 原文)
         │
         ▼  clean_document()        换行归一化 / 去零宽字符 / 压空行
     Document
         │
         ▼  split_document()        递归切分 + 贪心合并 + 重叠
      Chunk × N                     chunk_id = stable_id(doc_id, index, text)
         │
         ▼  embed_all()             分片 + 并发 + 保序，429/5xx 自动重试
     向量 × N
         │
         ▼  ingest_chunks()         先按 doc_id 删旧 → upsert → flush
   ┌─────────────┐
   │    Milvus   │  chunk_id / doc_id / text / chunk_index / embedding / sparse(BM25)
   └─────────────┘
         │
   query │
         ▼  Retriever.retrieve()
  ① 召回：稠密 search  或  dense + BM25 两路 hybrid_search(RRF)
  ② 精排：rerank_candidates 条候选 → cross-encoder → top_k
         │
         ▼
   list[ScoredChunk]
```

---

## API 一览

### 门面

```python
Ragkit(settings=None, *, embedder=None, indexer=None, reranker=None, splitter=None)

    .ingest(paths, *, reset=False, flush=True) -> IngestResult
    .query(text, *, top_k, mode, rerank, rerank_candidates,
                        score_threshold, filter_expr, total_timeout) -> list[ScoredChunk]
    .delete_document(doc_id) -> int
    .count() -> int
    .drop() -> None
    .aclose() -> None               # 也支持 async with
```

### 各层

| 层 | 入口 | 关键点 |
|---|---|---|
| 解析 | `parse_file()` / `parse_files()` | 异步；异常在边界翻译成 `ParseError` |
| 处理 | `clean_document()` / `normalize_whitespace()` | **同步**（纯 CPU，不等待） |
| 切分 | `split_document()` / `RecursiveSplitter` / `FixedSplitter` / `write_chunks_jsonl()` | **同步**；不变量 `"".join(pieces) == 原文` |
| 向量化 | `create_embedder()` / `embed_all()` / `HashingEmbedder` | `Embedder` 协议；返回顺序严格对应输入 |
| 索引 | `MilvusIndexer` / `ingest_chunks()` / `build_schema()` | 懒建表双检锁；写入后 flush |
| 检索 | `Retriever` / `doc_filter()` / `SectionAwareRetriever` | 分数方向见[查询](#查询) |
| 策略 | `STRATEGIES` / `resolve_strategy()` / `describe_strategies()` | 给「哪一层 + 什么模式 + 要不要重排」的组合起名字，配置里按名字选用（见 [MCP 配置](#配置)） |
| 重排 | `create_reranker()` / `HttpReranker` | 换 `score` 语义；候选池必须放大 |
| 评估 | `evaluate()` / `load_dataset()` / `dump_dataset()` / `hit_rate()` … | 容错跑批；支持按 `metadata` 分组 |

---

## 配置参考

所有配置走环境变量 / `.env`，由 `pydantic-settings` 读取（[`src/ragkit/config.py`](src/ragkit/config.py)）。

### 模型

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_CHAT_BASE_URL` | `https://api.deepseek.com` | 生成环节（LLM）用。**那一环还没实现**，这几个字段现在填不填都不影响 |
| `OPENAI_CHAT_API_KEY` | — | |
| `OPENAI_CHAT_MODEL` | `deepseek-flash` | ⚠️ 要填服务商**真实的**模型名（DeepSeek 官方是 `deepseek-chat`），否则调用时 404 |
| `LLM_TEMPERATURE` | `0.0` | |
| `OPENAI_EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | 任何 OpenAI 兼容端点 |
| `OPENAI_EMBEDDING_API_KEY` | — | 用 `SecretStr` 存，日志里显示成 `**********` |
| `OPENAI_EMBEDDING_MODEL` | `BAAI/bge-m3` | |
| `OPENAI_EMBEDDING_DIM` | `1024` | ⚠️ **必须和模型真实维度一致** |
| `OPENAI_EMBEDDING_BATCH_SIZE` | `32` | 单次请求塞多少条文本 |
| `RERANK_BASE_URL` | `https://api.siliconflow.cn/v1` | rerank **不是** OpenAI 协议的一部分 |
| `RERANK_API_KEY` | — | |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | |

### 存储与切分

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | |
| `MILVUS_TOKEN` | 空 | Zilliz Cloud 需要 |
| `MILVUS_DB` | 空（= default 库） | ⚠️ 库必须**先手动建好**，Milvus 不会自动创建；换库不会迁移已有数据 |
| `MILVUS_COLLECTION` | `ragkit_chunks` | |
| `MILVUS_BATCH_SIZE` | `500` | 单次 upsert 多少条 |
| `CHUNK_SIZE` | `800` | 每个 chunk 的目标字符数 |
| `CHUNK_OVERLAP` | `120` | 相邻 chunk 的重叠字符数 |
| `TOP_K` | `5` | 默认检索条数 |

### 运行时

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAX_CONCURRENCY` | `8` | 并发上限（HTTP 请求数 / 数据库写入批次） |
| `REQUEST_TIMEOUT` | `60.0` | 单次请求超时（秒） |
| `MAX_RETRIES` | `3` | 重试次数（**只重试** 429 / 5xx / 超时） |
| `RETRY_BASE_DELAY` | `1.0` | 指数退避基数：1s、2s、4s… |

> ⚠️ `extra="ignore"` 意味着**环境变量拼错一个字母不会报错**，只会静默用默认值。
> 联调「配置怎么没生效」时，第一个要查的就是键名拼写。

---

## 跑一遍完整示例

仓库带了一份示例文档（[`examples/sample_doc.md`](examples/sample_doc.md)，虚构的员工手册）和两个脚本：

```bash
uv run python scripts/ingest_demo.py examples/sample_doc.md          # 端到端：解析→切分→向量化→入库→检索对比
uv run python scripts/ingest_demo.py examples/sample_doc.md --reset  # schema 变了（比如刚加了 BM25）时先重置
uv run python scripts/ingest_demo.py examples/sample_doc.md --fake   # 假向量：不联网不花钱，只验证数据库那段
uv run python scripts/eval_demo.py examples/sample_doc.md --rerank   # 评估：稠密/混合/重排三路对比，带分组指标
```

---

## 命令行（CLI）

```
uv run ragkit --help
```

CLI 是这套工具的入场券：**不入库就没有东西可检索**。它和 MCP 的分工是
「**写的走 CLI，读的走 MCP**」—— 入库、切分、评估这些会改变状态的动作由人显式触发，
检索则作为只读工具交给 agent 随时调用。但 `query` **仍然留在 CLI**：
调试时要能脱离 agent 手动跑一次。「模型说搜不到」的时候，
第一件事是自己在终端敲一遍，看是检索真的坏了，还是模型用错了。

### 命令总览

| 命令 | 用途 | 碰数据库 | 主要开关 |
|---|---|---|---|
| `ragkit parse <paths...>` | 只看解析结果和字数 | 否 | `--json` |
| `ragkit split <path>` | 切分并导出，肉眼检查切得好不好 | 否 | `--out` ＋ 切分开关 |
| `ragkit ingest <paths...>` | 解析 → 切分 → 向量化 → 入库 | 是 | `--reset` `--no-flush` ＋ 切分开关 |
| `ragkit query <text>` | 终端里手动检索验证 | 是 | `-k` `--mode` `--rerank` `--filter` ＋ 检索开关 |
| `ragkit count` | 记录数（近似值） | 是 | `--json` |
| `ragkit info` | 当前生效的配置 + 库里的条数 | 是（读一次） | `--json` |
| `ragkit delete <doc_id...>` | 按 doc_id 删文档 | 是 | `--json` |
| `ragkit drop` | 删掉整张表（不可恢复） | 是 | `--yes` |
| `ragkit eval <dataset.jsonl>` | 在评估集上跑一遍，打印指标 | 是 | 和 `query` 同一组检索开关 |
| `ragkit mcp` | 起 MCP 服务端（stdio） | 是 | — |

切分开关 = `--splitter` `--chunk-size` `--chunk-overlap` `--no-context` `--no-merge`；
检索开关 = `--section-aware` `--candidate-factor` `--section-boost` `--max-per-section`。
两组都是「**同一组开关、多条命令共用**」，也就是 `split` 和 `ingest` 一定按同一套规则切，
`query` 和 `eval` 一定按同一套规则捞 —— 否则「本地先看看切得好不好」就失去了意义。

### 通用约定

**① 退出码分三类**（很多 CLI 一律 `exit 1`，脚本里就分不清「参数写错了」和「检索失败了」）：

| 退出码 | 含义 | 谁抛的 | 例子 |
|---|---|---|---|
| `0` | 成功 | — | 正常跑完 |
| `1` | 业务失败 | `RagkitError` 家族 | Milvus 连不上、embedding 超时、文件解析失败 |
| `2` | 参数错误 | argparse，或 CLI 的 `UsageError` | `--mode 乱填`、`--chunk-overlap 80 --chunk-size 50`、空查询 |
| `130` | 被 Ctrl+C 中断 | — | 手动打断 |

```bash
uv run ragkit query "问题" ; echo "退出码 $?"     # 1：检索失败
uv run ragkit query ""      ; echo "退出码 $?"     # 2：参数没给对
```

「参数之间」的约束（`overlap >= chunk_size`、`--section-boost -1`）由切分器 / 检索器的
构造函数校验，CLI 在边界上把它们重新分类成 `UsageError` —— 所以也是 `2`。

**② `--json`：所有命令都支持**，输出不转义的中文，给脚本用。

```bash
uv run ragkit count --json
uv run ragkit info --json                  # 全部配置项（密钥掩码），可直接 diff 两个环境
uv run ragkit query "问题" --json | ConvertFrom-Json
```

**③ `parse` / `ingest` 直接收目录**：递归子目录、只挑解析器认得的后缀
（`.md` `.txt` `.pdf` `.docx`）、按路径排序 —— 所以 bash 和 PowerShell 写法完全一致，
不用自己在 shell 里拼 glob。目录里一个支持的文件都没有时会直接报错，而不是静默地「0 个文件」。

**④ 配置读「当前工作目录」的 `.env`。** 在别的项目里调用这个 CLI 时这条最关键：

```bash
cd D:/other-project
uv run --project D:/400-ProgramFile/AIGC_CLASS/ragkit ragkit count   # 读的是 D:/other-project/.env
```

每个字段都有默认值，所以 `.env` 里**一个字段都可以不写**；但省略的后果分四类，得挑着省：

| 省略的字段 | 会怎样 |
|---|---|
| `*_API_KEY` | 空串 → 调用接口时才 401（响亮） |
| `OPENAI_EMBEDDING_URL/MODEL/DIM` | 走默认（硅基流动 + `BAAI/bge-m3` + 1024）；维度错了会在建表/检索时报错 |
| `MILVUS_URI` / `MILVUS_COLLECTION` | 走默认（`localhost:19530` + `ragkit_chunks`）→ **静默写到另一张表上** |
| `MILVUS_DB` | 空 = Milvus 的 `default` 库 → **连错库和「表是空的」长得一样** |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 走代码默认（800 / 120）→ 只要一处改过，两边的 `chunk_id` 就对不上（见[参数一致性](#参数一致性)） |
| `TOP_K` / 超时 / 批量 / 并发 | 无所谓，命令行参数还会覆盖 |

> **默认值来自代码，所以两边都不写 = 两边天然一致**；只有被显式写进某个 `.env` 的字段
> 才可能造成两边不一致 —— 半份复制才是最坏的情况。
> 键名拼错不会报错（`extra="ignore"`），只会被静默忽略：配完用 `ragkit info` 确认一遍，
> 它会打出最终生效的值和 `.env` 的绝对路径。
>
> 例外：**MCP 服务端固定读 ragkit 项目根的 `.env`**，不是 CWD 那份（见 [MCP 一节](#作为-mcp-服务端给-agent-用)）。
> 两边行为不一致时，先确认自己在看哪一个。

**⑤ 每个开关的默认值都写在 `--help` 里**，包括「这个默认值来自 `.env` 还是固定值」：
`uv run ragkit <命令> --help`。

### 命令详解

每个命令的**参数、默认值、以及它到底干什么**。`<...>` 是位置参数，其余是开关。

#### `parse <paths...> [--json]`

只走解析，不碰数据库、不需要 API key。

| 参数 | 默认 | 说明 |
|---|---|---|
| `paths` | 必填 | 文件**或目录**（可多个）；目录会递归展开并按路径排序 |
| `--json` | 关 | 输出 `[{source, chars, doc_id}]` |

```bash
uv run ragkit parse examples/sample_doc.md
#     2825 字  examples/sample_doc.md
#   1 个文件，共 2825 字
```

解析出来字数是 0 是最常见的意外（扫描件 PDF、加密 docx）—— 这一步就能看到，
不用等入库之后才发现「怎么什么都搜不到」。

#### `split <path> [--out f.jsonl] [切分开关]`

| 参数 | 默认 | 说明 |
|---|---|---|
| `path` | 必填 | **只接受单个文件**（要批量就先 `parse` 目录确认文件数） |
| `--out` | — | 导出 JSONL，每行一个 chunk（`chunk_id`/`doc_id`/`index`/`text`/`metadata`） |
| `--json` | 关 | 额外给出每个 chunk 的 `chunk_id` 和字符数 |
| 切分开关 | | 见[切分与检索的开关](#切分与检索的开关) |

```bash
# 完全离线：调切分参数时最常用的动作，不该被 Milvus 或 API key 卡住
uv run ragkit split examples/sample_doc.md --chunk-size 350 --chunk-overlap 50 --out chunks.jsonl
#   examples/sample_doc.md  ->  9 块  （min=315 max=397 平均=356，splitter=recursive chunk_size=350 overlap=50）

Get-Content chunks.jsonl | Select-Object -First 3     # PowerShell
head -n 3 chunks.jsonl                                # bash
```

#### `ingest <paths...> [--reset] [--no-flush] [切分开关]`

| 参数 | 默认 | 说明 |
|---|---|---|
| `paths` | 必填 | 文件或目录（可多个） |
| `--reset` | 关 | 先删掉整张表再建。**改过 schema 必须用**（字段建好之后改不了） |
| `--no-flush` | 关 | 跳过写入后的 flush：批量导入更快，代价是「写完立刻查」可能查不到 |
| `--json` | 关 | 输出 `files`/`chunks`/`written`/`doc_ids` ＋ `splitter`/`chunk_size`/`chunk_overlap` |
| 切分开关 | | 见[切分与检索的开关](#切分与检索的开关) |

```bash
uv run ragkit ingest examples/eng_docs/langchain --reset
# 入库完成：25 个文件  848 个 chunk  848 条记录
```

输出那三个数（`files` → `chunks` → `written`）本该相等；不等就说明中间某一步吞了东西
（有文件解析出来是空的，或写入阶段丢数据），CLI 会打一行 `⚠️` 提醒。
重复导入同一份文件是**幂等**的：会先按 `doc_id` 删掉旧数据，不会堆重复记录。

`--json` 里那三个切分参数是**事后唯一的凭据** —— 指标不对时，靠它才能证明
「入库和生成标注用的是不是同一套切分」（见[参数一致性](#参数一致性)）。

#### `query <text> [检索开关] [--filter EXPR]`

| 参数 | 默认 | 说明 |
|---|---|---|
| `text` | 必填 | 查询文本；空字符串算参数错误（退出码 2） |
| `-k` / `--top-k` | `TOP_K`（5） | 返回几条 |
| `--filter` | — | Milvus 过滤表达式，如 `doc_id == "abc123"`（只搜某几篇文档） |
| 检索开关 | | 见[切分与检索的开关](#切分与检索的开关) |

```bash
uv run ragkit query "住宿费上限" -k 3 --mode hybrid --rerank
uv run ragkit query "middleware" --section-aware --mode hybrid -k 3
uv run ragkit query "住宿费上限" --json | ConvertFrom-Json     # JSON 里带完整正文
```

输出是「分数 + 来源文件 + 前 70 字预览」。**分数的量纲随模式而变**：稠密是余弦
（0~1，越大越相关）、混合是 RRF 融合分（约 0~0.033）、开了重排又变成重排模型的相关度分
（0~1）—— 所以**跨模式比大小没有意义**，换一种检索方式，分数的尺子就换了。

「模型说搜不到」的时候，第一件事就是自己在这里敲一遍：看是检索真的坏了，
还是模型用错了（问法不对、该给 `filter`、或者它根本没调这个工具）。

#### `count` / `info` / `delete` / `drop` / `mcp`

| 命令 | 参数 | 干什么 |
|---|---|---|
| `count` | `--json` | 记录数（**近似值**，基于已落盘的 segment） |
| `info` | `--json` | 当前生效的配置 + 库里的条数 |
| `delete <doc_id...>` | `--json` | 按 doc_id 删文档，`doc_id` 从 `ingest` 的输出里拿 |
| `drop` | `--yes` | 删整张表（不可恢复）。默认要人工确认，确认框里显示「uri / 数据库 / collection」 |
| `mcp` | — | 起 MCP 服务端（stdio），阻塞到客户端断开或 Ctrl+C |

```bash
uv run ragkit info
# Milvus 地址：http://localhost:19530
# 数据库：agent_kb
# Collection：ragkit_chunks
# 向量维度：1024（OPENAI_EMBEDDING_DIM）
# Embedding：BAAI/bge-m3 @ https://api.siliconflow.cn/v1
# 重排模型：BAAI/bge-reranker-v2-m3（RERANK_API_KEY 已配 / 未配）
# 默认切分：chunk_size=800  chunk_overlap=120
# 默认检索：top_k=5（mode 由命令行的 --mode 决定）
# .env：D:\400-ProgramFile\AIGC_CLASS\ragkit\.env（存在）
#
# 记录数：848 条（近似值，刚写入还没 flush 的数据可能没算进来）
```

`info` 是排查的第一步 —— 它一次回答三个最容易搞混的问题：**连的是哪个库、表叫什么、
当前配置长什么样**（含 `.env` 的绝对路径）。连不上 Milvus 时它照样打印配置部分，
只有「记录数」那一行失败、退出码 1，而配置恰恰是那时候最需要看的。

`--json` 给的是**全部 24 项**（键名就是 `.env` 里的写法，密钥只显示「已配置 / 空」，
绝不回显），所以「A 项目和 B 项目的配置一致吗」可以直接 diff 两份 JSON；
**被省略的字段也在里面**（值就是代码默认值）——「谁被省略了」本身就是信息。

> ⚠️ 三个反直觉的点：
>
> 1. `count` / `info` 走的是 `ensure_collection()`，所以**表不存在时它会顺手建一个空表** ——
>    别以为「只读命令不会动数据库」。
> 2. `count` 是近似值。想知道「这次写进去多少条」，看 `ingest` 的 `written`。
> 3. `doc_id` 是「路径 + 内容」的哈希，**文件一改 id 就变**：想删旧数据不能靠重新解析
>    这个文件（算出来的 id 对不上），要么保留 `ingest` 返回的 `doc_ids`，要么 `drop` 后重来。

#### `eval <dataset.jsonl> [检索开关]`

| 参数 | 默认 | 说明 |
|---|---|---|
| `dataset` | 必填 | JSONL 评估集，每行 `{question, relevant_ids, metadata?}` |
| `--json` | 关 | 输出完整报告（含每条样本的结果与失败原因） |
| 检索开关 | | 和 `query` 共用同一套 —— 评估要测的就是你平时用的那套捞法 |

```bash
uv run ragkit eval examples/eval_sample.jsonl -k 3 --mode hybrid
```

打印 hit_rate / recall / mrr / ndcg。**打之前先看失败样本数** —— 把「接口超时」和
「检索质量差」混在一起会误判。它还会检查另一种情况：命中率为 0 却没有任何样本失败，
那几乎一定是标注和库里的数据对不上，CLI 会直接把这句话说出来。

#### `mcp`

```bash
uv sync --extra mcp      # mcp 是可选依赖，默认不装
uv run ragkit mcp        # 等价于 uv run python -m ragkit.mcp_server
```

配置见 [MCP 一节](#作为-mcp-服务端给-agent-用)。

### 切分与检索的开关

文档不一样，「怎么切」和「怎么捞」也该不一样。这两组开关**被多条命令共用**：
`split` 和 `ingest` 一定按同一套规则切，`query` 和 `eval` 一定按同一套规则捞 ——
否则「本地先看看切得好不好」就失去了意义。

| 开关 | 默认 | 作用 |
|---|---|---|
| `--splitter recursive` | ✅ | 按分隔符递归切，通用文本够用 |
| `--splitter section` | | 按 Markdown 标题分节，每块补 `[doc: X \| section: A > B]` 前缀；代码块不切开，frontmatter 剥掉 |
| `--chunk-size` | `.env` 的 `CHUNK_SIZE`（800） | 每块目标字符数 |
| `--chunk-overlap` | recursive：`.env` 的 `CHUNK_OVERLAP`（120）；**section：`0`** | 相邻块重叠字符数。两个切分器的默认值**刻意不同**，见下节 |
| `--no-context` | 关 | 只在 `section` 下生效：不补标题前缀（消融对照用） |
| `--no-merge` | 关 | 只在 `section` 下生效：不合并相邻小节（消融对照用） |
| `--section-aware` | 关 | 检索后处理：标题加权 + 去近似重复 + 同节限流 |
| `--candidate-factor` | `4` | 只在 `--section-aware` 下生效：先取 `top_k × 4` 条候选再做后处理 |
| `--section-boost` | `0.5` | 只在 `--section-aware` 下生效：标题全中时分数最多乘 1.5 |
| `--max-per-section` | `2` | 只在 `--section-aware` 下生效：同一节最多贡献几条 |

只在某个模式下生效的开关，如果**在别的模式下传了**，CLI 会在 stderr 上打一行提示说
「本次已忽略」——「参数传了却不生效」是这个项目里最难自查的一类问题，所以宁可啰嗦。

`--section-aware` **依赖 `[doc | section]` 前缀**（标题加权全靠那一行），所以要配
`--splitter section` 才有意义；挂在 recursive 的结果上，它只剩「去重 + 同节限流」，
而那两件事会退化成「每篇文档最多 2 条」。

### 参数一致性

`chunk_id` 是**切分结果**的内容哈希，由三个值共同决定：**切分器 + `chunk_size` +
`chunk_overlap`**。任意一个变了，整批 id 就全变 —— 而评估集里的标注（`relevant_ids`）
绑的正是 `chunk_id`。

两套 id 对不上时，检索**完全正常**，只是拿到的 id 一个也对不上：
指标悄悄偏低，一句报错都没有。实测（25 篇语料，标注按 `section(800/0)` 生成）：

| 入库用的切分 | 库里 chunk 数 | 与标注的交集 |
|---|---|---|
| `section(800/0)` | 1111 | **28 / 28** |
| `section(800/120)` | 1111 | 24 / 28 |

第二行就是「忘了调 overlap」的样子：不是全 0，而是**静悄悄地漏掉 14% 的标注**，
召回率上限从 1.0 掉到 0.857 —— 而你还以为是检索质量差。

三道防线：**① `section` 的默认 overlap 就是 0**（和生成脚本的默认值对齐）；
**② `ingest --json` 记下那三个值**（事后有据可查）；
**③ `eval` 在「命中率 0 且无失败样本」时直接告诉你**是标注对不上，不是检索坏了。
再自查就一条：`ragkit info` 打当前配置 + 库里条数，把「入库时记下的三个值」和
「生成评估集时用的三个值」比一比，问题立刻就出来了。

### 四组对照实验

想知道「换个切分 / 加个后处理到底值多少分」，`examples/eng_docs/` 下已经备好三份
对应的评估集（`section` 的 overlap 默认就是 0，不用再手写 `--chunk-overlap 0`）：

```bash
# A 基线：recursive 切分
uv run ragkit ingest examples/eng_docs/langchain --reset
uv run ragkit eval examples/eng_docs/eval_langchain_recursive.jsonl -k 3 --mode hybrid

# B 只换切分器：按标题分节 + 补标题路径
uv run ragkit ingest examples/eng_docs/langchain --splitter section --reset
uv run ragkit eval examples/eng_docs/eval_langchain_section.jsonl -k 3 --mode hybrid

# C 再叠结构感知检索（复用 B 已入库的数据，不必重新 ingest）
uv run ragkit eval examples/eng_docs/eval_langchain_section.jsonl -k 3 --mode hybrid --section-aware

# D 消融：关掉标题前缀，看「补标题」本身值多少
uv run ragkit ingest examples/eng_docs/langchain --splitter section --no-context --reset
uv run ragkit eval examples/eng_docs/eval_langchain_nocontext.jsonl -k 3 --mode hybrid
```

四组的 `chunk_id` 各不相同，所以评估集也要各一份：换语料或换参数时用
[`scripts/build_eval_variants.py`](scripts/build_eval_variants.py) 重新生成
（默认值和 `ragkit ingest` 完全一致，所以同一个配置两边算出的 `chunk_id` 必然对得上）。
脚本还会顺手检查「依据有没有命中」（没命中直接报错退出）和
「单条依据命中的块是不是太多」（超过 `--max-hits` 就告警 —— 依据太泛会让指标虚高）。

### 排查手册

按「最可能先撞上」排序。每一条都能用 CLI 自己查：

| 症状 | 最可能的原因 | 怎么办 |
|---|---|---|
| 指标全是 0，且没有失败样本 | 标注和库里不是同一套切分 | `ragkit info` 看配置 → 核对 `ingest --json` 里的三个值 → 用脚本按当前配置重生成评估集 |
| 指标低得莫名其妙 | 同上，只是漏了一部分标注 | 同上（`section` 忘了调 overlap 就是 28 → 24 那种） |
| 刚入库却查不到 | Milvus 是最终一致性 | 别加 `--no-flush`；或者稍等再 `count` |
| `count` 和 `written` 对不上 | `count` 是近似值 | 以 `ingest` 返回的 `written` 为准 |
| 换了 embedding 模型就报维度错 | `OPENAI_EMBEDDING_DIM` 没跟着改 | 改 `.env` 后 `ingest --reset`（schema 里的维度改不了） |
| 混合检索 + 阈值直接报错 | RRF 分（0~0.033）和余弦（0~1）不是一个量纲 | 去掉阈值，或者开 `--rerank` |
| 开了 `--rerank` 却没变化 | 没配 key，或候选池太小 | 看 `ragkit info` 那行「RERANK_API_KEY」；候选池默认 `max(k*4, 20)` |
| `--no-context` / `--no-merge` 看不出区别 | 它们只对 `--splitter section` 有效 | CLI 会打提示；加上 `--splitter section` |
| `--max-per-section` 之类看不出区别 | 没开 `--section-aware` | CLI 会打提示；加上 `--section-aware` |
| 数据「不见了」/ 删错了库 | `.env` 读的是**当前工作目录**那一份 | `ragkit info` 最后一行会打出 `.env` 的实际路径 |

---

## 作为 MCP 服务端给 agent 用

ragkit 可以把自己注册成一个 MCP 工具服务器，让本地 agent 直接检索你的知识库。

```bash
uv sync --extra mcp          # mcp 是可选依赖，默认不装
uv run python -m ragkit.mcp_server
```

**对外只有一个工具**：`search_knowledge_base(query, top_k)`，只读。

- **只有只读检索。** 这些工具会被 agent 自动注册给模型，而**模型能调什么，就等于它能做什么**。
  入库、删除、重建 collection 都是运维动作，留在 CLI 里 —— 否则模型一句「帮我清理一下」
  就能把库删了。
- **标了 `read_only_hint=True`。** 客户端据此判断要不要人工审批；不标的话，纯读的检索
  也会每次弹确认，在非交互场景（脚本、CI）里会被直接拒绝。
- **返回文本、不带分数。** 混合检索的分是 RRF（0~0.033），加了重排又变成 0~1，同一个字段
  两种量纲，模型解读不了 —— 实测会让它得出「匹配度很低」这种错误结论。
  同理，**用哪个策略也只写进日志**，那种诊断信息一旦进了上下文，模型很容易把它
  写进给用户的答案里（「我用 hybrid_rerank 检索到……」）。
- **`top_k` 上限 10。** 模型偶尔会要 100 条，那会把上下文塞爆。

### 配置

MCP 服务端**只有三个配置项**，都走环境变量（**不是**工具参数）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAGKIT_MCP_STRATEGY` | `hybrid_rerank` | 用哪种检索方式，取值见下表 |
| `RAGKIT_MCP_MODE` | — | ⚠️ 已废弃，改用上面那个 |
| `RAGKIT_MCP_RERANK` | — | ⚠️ 已废弃；仍然认，会映射成策略名并打一行警告 |

**两个来源都认，优先级是「进程环境变量 > 项目根的 `.env`」**（和 CLI 的规则一致）：

```ini
# ① 项目根的 .env —— 本机改一下，所有客户端都跟着变
RAGKIT_MCP_STRATEGY=section_aware
```

```json
// ② MCP 客户端的 env 块 —— 优先级更高，只影响这一个客户端
"env": { "RAGKIT_MCP_STRATEGY": "section_aware" }
```

> ⚠️ 注意路径：CLI 读**当前工作目录**的 `.env`，而 MCP 服务端固定读 **ragkit 项目根**
> 的那份（它由 agent 拉起，工作目录不可控）。两边行为不一致时，先确认自己在看哪一个。

**检索方式不给模型选** —— 工具上没有这个参数。模型既不知道这套库上哪种更好，也不知道
你有没有配重排 key；给它一个参数，结果就是每次随机试一种，而且它会拿着「这次换策略了」
的错觉去解读同一批结果（换一种策略，分数的尺子都变了）。要换就改配置，改一次就一直是它。

| 策略 | 组成 | 擅长 |
|---|---|---|
| `dense` | 纯向量 | 「换一种说法问同一件事」；最快最省 |
| `hybrid` | 向量 + BM25（RRF 融合） | 含专有名词、编号、罕见词的提问 |
| `hybrid_rerank` | hybrid + 重排 | 提问用词和原文差得很远（**默认**） |
| `section_aware` | hybrid + 结构感知后处理 | 同一主题的工程文档（依赖 `[doc \| section]` 前缀，要 `--splitter section` 入库过） |
| `section_aware_rerank` | 上面 + 重排 | 工程文档语料上通常是最强的一档 |

**名字写错会在启动时直接报错**，并把可选值连同「擅长什么」一起列出来 —— 服务起不来，
你当场就知道；这比「检索结果悄悄变差」好查得多。

> 没配 `RERANK_API_KEY` 时**不要**用带 `rerank` 的策略，否则每次检索都会失败；
> `ragkit info` 里能看到这个 key 配没配。

### 客户端配置

```json
{
  "mcpServers": {
    "ragkit": {
      "command": "D:/path/to/ragkit/.venv/Scripts/python.exe",
      "args": ["-m", "ragkit.mcp_server"],
      "cwd": "D:/path/to/ragkit",
      "env": { "RAGKIT_MCP_STRATEGY": "hybrid_rerank" }
    }
  }
}
```

**直接调 venv 里的 python，不要用 `uv run`。** 两个原因：

- `uv run` 每次启动都要做一遍依赖解析检查，慢；
- 如果父进程已经设了 `VIRTUAL_ENV`（很多工具会这么干），`uv run` 会打一行
  「`VIRTUAL_ENV` does not match the project environment path」的警告。
  它走 stderr、不影响协议，但每次启动都刷一遍很烦。

> **`cwd` 建议显式写上。** 服务端找 `.env` 用的是**项目根的固定路径**（不依赖工作目录），
> 所以不填也能跑；写上是为了让「这个进程在哪、读的哪个项目」一目了然。
>
> 换 `uv` 也行：`{"command": "uv", "args": ["run", "--extra", "mcp", "python", "-m", "ragkit.mcp_server"]}`，
> 代价是多一次依赖解析和可能的警告。

**验证配置生效了没有**：启动后正常是**一句话都不输出**（stdout 是协议通道）；
如果策略名写错了，进程会立刻退出并打印一行「不认识 …」＋全部可选值。
另外每次检索都会往 stderr 打一行日志：

```
检索完成：strategy=section_aware top_k=5 命中=3
```

### 调试注意

**stdio 模式下 stdout 是 JSON-RPC 协议通道。** 任何 `print()` 都会污染协议流，
让客户端解析失败，而且报错完全指不到原因。调试信息一律走 `logging`（默认输出到 stderr）。

> 顺带一句：MCP SDK 导入时会把 root logger 设成 INFO，handler 指向 **stderr**，
> 所以默认是安全的；但 httpx 会跟着把每一次请求都打出来。
> [`mcp_server.py`](src/ragkit/mcp_server.py) 里把 `httpx` 压到了 WARNING ——
> 排查「搜不到」的时候，看我们自己那行「检索完成：strategy=… 命中=…」就够了。

---

## 评估结果

用 `examples/sample_doc.md`（2825 字 → 9 块）+ 11 个问题（关键词自动标注），`top_k=3`：

```
指标              稠密        混合      混合+重排
hit_rate        0.909      1.000      1.000
recall          0.909      1.000      1.000
mrr             0.864      0.955      1.000
ndcg            0.876      0.966      1.000
```

> 📌 这份评估集是**关键词自动标注**的，对 BM25 天然友好。它适合跑通流程，不适合下结论。
> 严谨做法是人工标注 + 冻结 JSONL，并且**评估集要和切分策略一起版本化**
> （改了 `CHUNK_SIZE`，`chunk_id` 全变，标注就失效了）。

---

## 常见坑速查

按「最常踩」排序，每条都对应代码里的一处显式处理。**操作层面**的坑（连错库、写完查不到、
指标异常）在[排查手册](#排查手册)里，这里只列**写代码时**容易踩的：

1. **`OPENAI_EMBEDDING_DIM` 写错** → 不会立刻报错，会一路走到建 Milvus 表才失败。代码里设了三道校验挡它。
2. **Milvus 版本低于 2.5** → `hybrid_search` / `RRFRanker` 不存在。启动时先打印服务端版本。
3. **`x or default` 的 falsy 陷阱** → `top_k=0` 会被静默换成默认值，让校验形同虚设。用 `x if x is not None else default`。
4. **只把 top-k 拿去重排** → 重排只能在 k 条里换顺序，等于白开。候选池必须放大（默认 `max(k*4, 20)`）。
5. **混合模式设 `score_threshold`** → 直接报错，因为 RRF 分（0~0.033）和余弦（0~1）不是一个量纲。
6. **改了 schema 没重建 collection** → 字段是改不了的，必须 `reset=True` / `--reset` / `drop()` 后重来。
7. **`except ParseError` 抓不到 TaskGroup 的异常** → 它被包在 `ExceptionGroup` 里；`Ragkit` 会自动解包单个错误。
8. **`ruff format .` 把语料改了** → `examples/eng_docs/` 是数据不是代码，里面的代码块一旦被重排，
   `chunk_id` 全变、评估标注集体失效。`pyproject.toml` 里已经用 `extend-exclude` 把它排除掉了。

---

## 开发

```bash
uv run ruff format .      # 排版
uv run ruff check .       # 静态检查
uv run mypy src           # 类型检查（当前 0 错误）
uv run pytest -q          # 281 条测试，全部离线
```

**测试不需要 Milvus，也不需要 API key**，靠的是三件事：

- `httpx.MockTransport` 替换 HTTP 传输层；
- `VectorStore` / `Embedder` / `Reranker` 三个协议让替身只有几十行；
- `HashingEmbedder` 提供确定性的本地向量。

### 项目结构

```
src/ragkit/
├── config.py          Settings（pydantic-settings）
├── schemas.py         Document / Chunk / ScoredChunk / RAGResult
├── errors.py          异常体系（每种失败一个类型）
├── http.py            异步 JSON 客户端（重试 / 限流 / 异常翻译）
├── parsing/           文件 → Document
├── processing.py      清洗（同步，纯 CPU）
├── splitting/         Document → Chunk（同步；含 MarkdownSectionSplitter）
├── embedding/         文本 → 向量
├── indexing/          chunk + 向量 → Milvus
├── retrieval/         查询 → ScoredChunk（稠密/混合/重排 + SectionAwareRetriever）
├── evaluation/        HitRate / Recall / MRR / NDCG
├── pipeline.py        Ragkit 门面
├── cli.py             命令行（parse / split / ingest / query / count / info / …）
└── mcp_server.py      MCP 服务端（只读检索）

tests/                 281 条测试，离线
scripts/               端到端 demo、评估对比、评估集生成
examples/              sample_doc.md，以及 eng_docs/（LangChain 工程文档语料）
```

---

## 已知限制

| 限制 | 说明 |
|---|---|
| **没有生成环节** | 只做检索，不调 LLM 出答案。`RAGResult` 模型已就位，接上即可 |
| **CI 里没有真 Milvus** | 281 条测试全离线，证明**逻辑**正确，证明不了**契约**。见「设计笔记」第 5 条 |
| **MCP 层只测了格式化与策略选择** | 真正跑一次检索要连 Milvus，属于集成测试的范畴 |
| **测试替身有三份重复** | `FakeMilvusClient` 手抄了三份并已漂移。正式落点已定为 [`tests/fakes.py`](tests/fakes.py)，存量迁移待办 |
| **自动标注的评估集** | 见[评估结果](#评估结果)，它偏向关键词匹配 |
| **`Ragkit` 没暴露 `flush()`** | 大批量导入想手动控制刷盘时机，得用底层的 `MilvusIndexer` |
| **PDF 不支持扫描件** | 抽出空文本会显式报错，而不是静默产出空 chunk。需要 OCR 得另接 |
| **没有 LICENSE** | 要发布先补上 |

---

## 设计笔记

几条贯穿全项目的原则，写在这里免得后来的人（包括三个月后的你）重新踩一遍。

**1. 判断 async 的标准是「会不会等待」，不是「活儿重不重」。**

解析要读磁盘 → `async`；切分只算字符串 → **保持同步**。
给纯 CPU 的函数套 `async`，调用方被迫写 `await`，并发收益却是零。

**2. 底层异常必须在边界翻译成自己的类型。**

用户不该为了用这个库去 `import pymilvus` 或 `import httpx`。
`except RagkitError` 应该能兜住一切。`TaskGroup` 的 `ExceptionGroup` 也不例外。

**3. `x or default` 只在「空值和没提供是一回事」时成立。**

`batch_size=0`、`top_k=0` 都是**非法值**而不是「没提供」，
写 `or` 会把它们静默换成默认值，让校验形同虚设。

**4. 谁创建，谁销毁。**

每个组合类（`MilvusIndexer` / `Retriever` / `Ragkit`）都记着哪些组件是自己建的。
注入进来的绝不关——那是别人的资源。

**5. mock 能验证逻辑，验证不了契约。**

这个项目里有两个 bug 是**只有真环境才能发现**的：
稠密检索漏传 `anns_field`（假客户端不校验），
写入后没 `flush`（假客户端不模拟最终一致性）。
所以 `scripts/ingest_demo.py` 这种端到端脚本要留着——
**离线测试给你信心，端到端给你真相。**
