# ragkit

> 一个「能用、能懂、能改」的通用 RAG 工具包：**文件解析 → 清洗切分 → 向量化 → Milvus 索引 → 检索（稠密 / 混合 / 重排）→ 评估**。

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-200%20passed-brightgreen)
![Async](https://img.shields.io/badge/asyncio-first-orange)
![Vector%20DB](https://img.shields.io/badge/Milvus-2.5%2B-00A1EA)

---

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [工作流程](#工作流程)
- [API 一览](#api-一览)
- [配置](#配置)
- [跑一遍完整示例](#跑一遍完整示例)
- [评估结果](#评估结果)
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
| **混合检索** | 稠密向量 + BM25 关键词，RRF 融合 —— 专治「PPL」「E-204」这类稀有关键词 |
| **重排（Rerank）** | 两阶段检索：召回 20~50 条候选 → cross-encoder 精排取 top-k |
| **可评估** | HitRate / Recall / MRR / NDCG，支持**按问题分组**看指标、跑批容错 |
| **可替换** | 解析器 / 切分器 / Embedder / 向量库 / 重排器全是 Protocol，换实现不改上层 |

---

## 快速开始

### 1. 安装

```bash
git clone <repo> && cd ragkit
uv sync
```

### 2. 配置

复制 `.env.example` 为 `.env`，填上你的 key：

```ini
# ---- 生成（当前未接入，见「已知限制」）----
OPENAI_CHAT_BASE_URL=https://api.deepseek.com
OPENAI_CHAT_API_KEY=sk-...
OPENAI_CHAT_MODEL=deepseek-chat

# ---- 向量化：任何 OpenAI 兼容端点 ----
OPENAI_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_EMBEDDING_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=BAAI/bge-m3
OPENAI_EMBEDDING_DIM=1024          # ⚠️ 必须和模型的真实维度一致

# ---- 重排：注意 rerank 不是 OpenAI 协议的一部分 ----
RERANK_BASE_URL=https://api.siliconflow.cn/v1
RERANK_API_KEY=sk-...
RERANK_MODEL=BAAI/bge-reranker-v2-m3

# ---- Milvus（需要 2.5+ 才有 BM25 / hybrid_search）----
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION=ragkit_chunks
```

### 3. 五行跑通

```python
import asyncio
from ragkit import Ragkit


async def main() -> None:
    async with Ragkit() as rag:
        await rag.ingest(["docs/员工手册.md", "docs/报销制度.pdf"])
        hits = await rag.query("出差住宿费上限是多少", mode="hybrid", rerank=True)
        for hit in hits:
            print(f"{hit.score:.3f}  {hit.chunk.text[:60]}")


asyncio.run(main())
```

`Ragkit` 是薄门面：它只负责**按顺序调用**和**管理生命周期**。
每个组件都能注入替换（`embedder=` / `indexer=` / `reranker=` / `splitter=`），
测试和定制都靠它。

---

## 工作流程

```
  .md / .pdf / .docx
         │
         ▼  parse_files()           并发解析，TaskGroup + Semaphore 限流
     Document                       doc_id = sha256(路径 + 原文)
         │
         ▼  clean_document()        换行归一化 / 去零宽字符 / 压空行
     Document
         │
         ▼  split_document()        递归切分 + 贪心合并 + 重叠
      Chunk × N                     chunk_id = sha256(doc_id, index, text)
         │
         ▼  embed_all()             分片 + 并发 + 保序，失败重试
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

### 门面层

```python
Ragkit(settings=None, *, embedder=..., indexer=..., reranker=..., splitter=...)
    .ingest(paths, reset=False, flush=True) -> IngestResult
    .query(text, *, top_k, mode, rerank, rerank_candidates, score_threshold, filter_expr)
    .count() -> int
    .drop() -> None
```

### 各层单独使用

| 层 | 入口 | 关键点 |
|---|---|---|
| 解析 | `parse_file()` / `parse_files()` | 异步；异常在边界翻译成 `ParseError` |
| 处理 | `clean_document()` / `normalize_whitespace()` | **同步**（纯 CPU，不等待） |
| 切分 | `split_document()` / `RecursiveSplitter` / `FixedSplitter` | **同步**；不变量 `"".join(pieces) == 原文` |
| 向量化 | `create_embedder()` / `embed_all()` | `Embedder` 协议；`HashingEmbedder` 可离线 |
| 索引 | `MilvusIndexer` / `ingest_chunks()` | 懒建表双检锁；写入后 flush |
| 检索 | `Retriever` / `doc_filter()` | 分数方向：稠密 COSINE 越大越好，混合 RRF 越大越好 |
| 重排 | `create_reranker()` / `HttpReranker` | 换 `score` 语义；候选池必须放大 |
| 评估 | `evaluate()` / `load_dataset()` | 容错跑批；支持按 `metadata` 分组 |

### 分数语义（很容易踩的坑）

| 阶段 | 分数含义 | 量级 |
|---|---|---|
| 稠密检索 | 余弦相似度 | 0 ~ 1 |
| 混合检索 | RRF 融合分 | 约 0 ~ 0.033 |
| **重排之后** | 重排模型的相关度分 | 0 ~ 1 |

三种**完全不是一个量纲**。所以：

- 混合模式下传 `score_threshold` 会**直接报错**，除非同时开了 `rerank`；
- 别把某个阶段调好的阈值套到另一个阶段。

---

## 配置

所有配置走环境变量 / `.env`，由 `pydantic-settings` 读取（[`src/ragkit/config.py`](src/ragkit/config.py)）。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAX_CONCURRENCY` | `8` | 并发上限（HTTP 请求数 / 数据库写入批次） |
| `REQUEST_TIMEOUT` | `60.0` | 单次请求超时（秒） |
| `MAX_RETRIES` | `3` | 重试次数（只重试 429 / 5xx / 超时） |
| `RETRY_BASE_DELAY` | `1.0` | 指数退避基数：1s、2s、4s… |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | 切分粒度 |
| `TOP_K` | `5` | 默认检索条数 |
| `OPENAI_EMBEDDING_BATCH_SIZE` | `32` | 单次 embedding 请求塞多少条 |
| `MILVUS_BATCH_SIZE` | `500` | 单次 upsert 多少条 |

> ⚠️ `extra="ignore"` 意味着**环境变量拼错一个字母不会报错**，只会静默用默认值。
> 联调「配置怎么没生效」时，第一个要查的就是键名拼写。

---

## 跑一遍完整示例

仓库里带了一份示例文档（[`examples/sample_doc.md`](examples/sample_doc.md)，虚构的员工手册）和两个脚本：

```bash
# 端到端：解析 → 切分 → 向量化 → 入库 → 检索对比
uv run python scripts/ingest_demo.py examples/sample_doc.md

# schema 变了（比如刚加了 BM25 稀疏字段）时先重置
uv run python scripts/ingest_demo.py examples/sample_doc.md --reset

# 不联网、不花钱：用本地假向量，只验证数据库那段通不通
uv run python scripts/ingest_demo.py examples/sample_doc.md --fake

# 评估：稠密 / 混合 /（--rerank）三路对比
uv run python scripts/eval_demo.py examples/sample_doc.md --rerank
```

---

## 评估结果

用 `examples/sample_doc.md`（2825 字，切成 9 块）+ 11 个问题（自动标注），`top_k=3`：

```
指标              稠密        混合      混合+重排
hit_rate        0.909      1.000      1.000
recall          0.909      1.000      1.000
mrr             0.864      0.955      1.000
ndcg            0.876      0.966      1.000

分组 NDCG（组内平均）
分组      题数     稠密        混合      混合+重排
直问        4   1.000      1.000      1.000
改述        4   0.908      0.908      1.000
稀有词      3   0.667      1.000      1.000
```

**读这张表的关键是分组那一列**：

- **混合检索的价值全在「稀有词」**（0.667 → 1.000）——BM25 补字面匹配；
- **重排的价值全在「改述」**（0.908 → 1.000）——cross-encoder 补语义排序，
  而混合在这一组毫无贡献；
- 综合分（+0.09）看着微不足道，分组之后才知道两块短板被分别补上了。

> 📌 这份评估集是**关键词自动标注**的，所以对 BM25 天然友好。
> 它适合跑通流程，不适合下结论。严谨做法是人工标注 + 冻结 JSONL，
> 而且**评估集要和切分策略一起版本化**（改了 `CHUNK_SIZE`，chunk_id 全变，标注就失效了）。

---

## 开发

```bash
uv run ruff format .      # 排版
uv run ruff check .       # 静态检查
uv run mypy src           # 类型检查
uv run pytest -q          # 200 条测试，全部离线
```

**测试不需要 Milvus，也不需要 API key。** 靠的是：

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

tests/                 200 条测试，离线
scripts/               端到端 demo 与评估
examples/              示例文档
```

---

## 已知限制

| 限制 | 说明 |
|---|---|
| **没有生成环节** | 只做检索，不调 LLM 出答案（原 M7 暂时搁置）。`RAGResult` 模型已就位，接上即可。 |
| **CI 里没有真 Milvus** | 200 条测试全是离线的，证明**逻辑**正确，证明不了**契约**。schema 变更、BM25 索引、最终一致性这些只有真环境能验——见「设计笔记」。 |
| **测试替身有三份重复** | `FakeMilvusClient` 手抄了三份并已开始漂移。正式落点已定为 [`tests/fakes.py`](tests/fakes.py)，存量迁移是待办。 |
| **自动标注的评估集** | 见上文，它偏向关键词匹配。 |
| **Milvus 2.5+** | BM25 / `hybrid_search` / `RRFRanker` 需要 2.5 以上；本仓库在 2.6.15 上验证。 |
| **PDF 不支持扫描件** | 抽出空文本会显式报错而不是静默产出空 chunk。需要 OCR 的话得另接。 |

---

## 设计笔记

几条贯穿全项目的原则，写在这里免得后来的人（包括三个月后的你）重新踩一遍。

**1. 判断 async 的标准是「会不会等待」，不是「活儿重不重」。**

解析要读磁盘 → `async`；切分只算字符串 → **保持同步**。
给纯 CPU 的函数套 `async`，调用方被迫写 `await`，并发收益却是零。

**2. 底层异常必须在边界翻译成自己的类型。**

用户不该为了用这个库去 `import pymilvus` 或 `import httpx`。
`except RagkitError` 应该能兜住一切。`TaskGroup` 的 `ExceptionGroup`
也不例外——门面在只有一个错误时会把它解包，因为 `except* ParseError` 太难懂。

**3. `x or default` 只在「空值和没提供是一回事」时成立。**

`batch_size=0`、`top_k=0` 都是**非法值**而不是「没提供」，
写 `or` 会把它们静默换成默认值，让校验形同虚设。用 `x if x is not None else default`。

**4. 谁创建，谁销毁。**

每个组合类（`MilvusIndexer` / `Retriever` / `Ragkit`）都记着哪些组件是自己建的。
注入进来的绝不关——那是别人的资源。

**5. mock 能验证逻辑，验证不了契约。**

这个项目里有两个 bug 是**只有真环境才能发现**的：
稠密检索漏传 `anns_field`（假客户端不校验），
写入后没 `flush`（假客户端不模拟最终一致性）。
所以 `scripts/ingest_demo.py` 这种端到端脚本要留着——
**离线测试给你信心，端到端给你真相。**

---

## 许可

本仓库目前未附带开源许可证文件。若要发布，请先补上 `LICENSE`。
