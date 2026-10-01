| 里程碑 | 内容 | 异步教学重点 |
|---|---|---|
| M0 | uv + git 初始化、工具链 | — |
| M1 | 数据模型 `Document`/`Chunk`/`ScoredChunk` + 配置 | `pydantic-settings` 读 `.env` |
| M2 | 文件解析（txt/md/pdf/docx） | `asyncio.to_thread`、`aiofiles`、`TaskGroup` |
| M3 | 清洗 + 切分策略 | 并发切分、背压 |
| M4 | Embedding（OpenAI 兼容） | `httpx.AsyncClient`、`Semaphore`、`gather`、指数退避重试 |
| M5 | Milvus 建表/写入/删除 | `AsyncMilvusClient`、批量并发写入 |
| M6 | 检索（dense + 过滤 + 可选 hybrid） | 超时控制、`asyncio.wait_for` |
| M7 | 简单生成（拼 prompt 调 LLM） | 流式 vs 非流式、`async for` |
| M8 | 评估（HitRate/MRR/忠实度） | 数据集并发跑 + 限流 |
| M9 | 门面 API + examples + README | — |