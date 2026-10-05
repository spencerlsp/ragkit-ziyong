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
| **递归切分** | 按语义层级切（段落 → 换行 → 句号 → 字符），保证 `"".join(pieces) == 原文` |
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
uv run pytest -q          # 200 条测试，全部离线，不需要 Milvus 和 API key
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

#### 2. 跑评估

```python
from ragkit import EvalSample, evaluate, load_dataset

samples = load_dataset("data/eval.jsonl")

async with Ragkit() as rag:
    dense = await evaluate(samples, rag._retriever, k=3, mode="dense")
    hybrid = await evaluate(samples, rag._retriever, k=3, mode="hybrid")

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
| 检索 | `Retriever` / `doc_filter()` | 分数方向见[查询](#查询) |
| 重排 | `create_reranker()` / `HttpReranker` | 换 `score` 语义；候选池必须放大 |
| 评估 | `evaluate()` / `load_dataset()` / `dump_dataset()` / `hit_rate()` … | 容错跑批；支持按 `metadata` 分组 |

---

## 配置参考

所有配置走环境变量 / `.env`，由 `pydantic-settings` 读取（[`src/ragkit/config.py`](src/ragkit/config.py)）。

### 模型

| 变量 | 默认值 | 说明 |
|---|---|---|
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
# 端到端：解析 → 切分 → 向量化 → 入库 → 稠密/混合检索对比
uv run python scripts/ingest_demo.py examples/sample_doc.md

# schema 变了（比如刚加了 BM25 稀疏字段）时先重置
uv run python scripts/ingest_demo.py examples/sample_doc.md --reset

# 不联网、不花钱：本地假向量，只验证数据库那段
uv run python scripts/ingest_demo.py examples/sample_doc.md --fake

# 换个验证问题
uv run python scripts/ingest_demo.py examples/sample_doc.md --query "报销多久到账"

# 评估：稠密 / 混合 /（--rerank）三路对比，带分组指标
uv run python scripts/eval_demo.py examples/sample_doc.md --rerank
```

---

## 作为 MCP 服务端给 agent 用

ragkit 可以把自己注册成一个 MCP 工具服务器，让本地 agent 直接检索你的知识库。

```bash
uv sync --extra mcp          # mcp 是可选依赖，默认不装
uv run python -m ragkit.mcp_server
```

**只暴露了一个工具**：`search_knowledge_base(query, top_k)`。

**为什么只有只读检索** —— 这些工具会被 agent 自动注册给模型，而**模型能调什么，就等于它能做什么**。
入库、删除、重建 collection 都是运维动作，留在 CLI 里；否则模型一句「帮我清理一下」就能把库删了。

**返回文本、不带分数** —— 混合检索的分是 RRF（0~0.033），加了重排又变成 0~1，
同一个字段两种量纲，模型解读不了，实测会让它得出「匹配度很低」这种错误结论。分数是给调试用的。

### 客户端配置

```json
{
  "mcpServers": {
    "ragkit": {
      "command": "D:/path/to/ragkit/.venv/Scripts/python.exe",
      "args": ["-m", "ragkit.mcp_server"],
      "cwd": "D:/path/to/ragkit"
    }
  }
}
```

**直接调 venv 里的 python，不要用 `uv run`。** 两个原因：

- `uv run` 每次启动都要做一遍依赖解析检查，慢；
- 如果父进程已经设了 `VIRTUAL_ENV`（很多工具会这么干），`uv run` 会打一行
  「`VIRTUAL_ENV` does not match the project environment path」的警告。
  它走 stderr、不影响协议，但每次启动都刷一遍很烦。

> ⚠️ **`cwd` 一定要填。** `.env` 是按工作目录找的，而 agent 拉起子进程时的工作目录由客户端决定。
> 代码里已经兜了一层（会去项目根找 `.env`），但显式写上最保险。
>
> 如果更习惯 `uv`，`{"command": "uv", "args": ["run", "--extra", "mcp", "python", "-m", "ragkit.mcp_server"]}` 也能用，只是会多一次解析和可能的警告。

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAGKIT_MCP_MODE` | `hybrid` | `dense` / `hybrid`；写错会在**启动时**直接报错 |
| `RAGKIT_MCP_RERANK` | `1` | 设 `0` 关闭重排。**没配 `RERANK_API_KEY` 时必须设成 0**，否则每次检索都失败 |

> 检索策略走环境变量而不是工具参数，因为它属于**部署决策** ——
> 不该让模型每次现挑（它没有判断依据，只会随便试）。

### 调试注意

**stdio 模式下 stdout 是 JSON-RPC 协议通道。** 任何 `print()` 都会污染协议流，
让客户端解析失败，而且报错完全指不到原因。调试信息一律走 `logging`（默认输出到 stderr）。

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

按「最常踩」排序。每条都对应代码里的一处显式处理，不是空话。

1. **`OPENAI_EMBEDDING_DIM` 写错** → 不会立刻报错，会一路走到建 Milvus 表才失败。代码里设了三道校验挡它。
2. **Milvus 版本低于 2.5** → `hybrid_search` / `RRFRanker` 不存在。启动时先打印服务端版本。
3. **写入后立刻检索查不到** → Milvus 是**最终一致性**的，新数据要先落盘。`ingest` 默认会自动 `flush`。
4. **混合模式设 `score_threshold`** → 会直接报错，因为 RRF 分（0~0.033）和余弦相似度（0~1）不是一个量纲。
5. **只把 top-k 拿去重排** → 重排只能在 k 条里换顺序，等于白开。候选池必须放大（默认 `max(k*4, 20)`）。
6. **`x or default` 的 falsy 陷阱** → `top_k=0` 会被静默换成默认值，让校验形同虚设。用 `x if x is not None else default`。
7. **改了 schema 没重建 collection** → 字段是改不了的，必须 `reset=True` 或 `drop()` 后重来。
8. **环境变量拼错** → `extra="ignore"` 会静默忽略，用默认值跑。
9. **`except ParseError` 抓不到 TaskGroup 的异常** → 它被包在 `ExceptionGroup` 里；`Ragkit` 会自动解包单个错误。
10. **文件改了想删旧数据** → `doc_id` 是内容哈希，改内容就换了身份，只能整表重建或保存原始 `doc_ids`。

---

## 开发

```bash
uv run ruff format .      # 排版
uv run ruff check .       # 静态检查
uv run mypy src           # 类型检查（当前 0 错误）
uv run pytest -q          # 206 条测试，全部离线
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
├── splitting/         Document → Chunk（同步）
├── embedding/         文本 → 向量
├── indexing/          chunk + 向量 → Milvus
├── retrieval/         查询 → ScoredChunk（含混合检索与重排）
├── evaluation/        HitRate / Recall / MRR / NDCG
└── pipeline.py        Ragkit 门面

tests/                 206 条测试，离线
scripts/               端到端 demo 与评估
examples/              示例文档
```

---

## 已知限制

| 限制 | 说明 |
|---|---|
| **没有生成环节** | 只做检索，不调 LLM 出答案。`RAGResult` 模型已就位，接上即可 |
| **CI 里没有真 Milvus** | 206 条测试全离线，证明**逻辑**正确，证明不了**契约**。见「设计笔记」第 5 条 |
| **MCP 层只测了格式化函数** | `search_knowledge_base` 要连真 Milvus，属于集成测试范畴 |
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
