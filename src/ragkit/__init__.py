"""ragkit —— 通用 RAG 工具包。

对外暴露的公共 API 都在这里 re-export，使用者只需要 ``import ragkit``。

"""

from .config import Settings, get_settings
from .embedding import Embedder, create_embedder, embed_all
from .errors import (
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IndexingError,
    ParseError,
    RagkitError,
    RetrievalError,
    SplitError,
)
from .evaluation import EvalReport, EvalSample, evaluate, load_dataset
from .indexing import MilvusIndexer, build_schema, ingest_chunks
from .parsing import parse_file, parse_files
from .pipeline import IngestResult, Ragkit
from .processing import clean_document, normalize_whitespace
from .retrieval import HttpReranker, Reranker, Retriever, create_reranker, doc_filter
from .schemas import Chunk, Document, RAGResult, ScoredChunk
from .splitting import (
    FixedSplitter,
    RecursiveSplitter,
    Splitter,
    split_document,
    split_documents,
    write_chunks_jsonl,
)

__all__ = [
    "parse_file",
    "parse_files",
    "Ragkit",
    "IngestResult",
    "clean_document",
    "normalize_whitespace",
    "split_document",
    "split_documents",
    "Splitter",
    "RecursiveSplitter",
    "FixedSplitter",
    "write_chunks_jsonl",
    "Document",
    "Chunk",
    "ScoredChunk",
    "RAGResult",
    "Settings",
    "get_settings",
    "Embedder",
    "create_embedder",
    "embed_all",
    "MilvusIndexer",
    "ingest_chunks",
    "build_schema",
    "Retriever",
    "doc_filter",
    "Reranker",
    "HttpReranker",
    "create_reranker",
    "evaluate",
    "EvalSample",
    "EvalReport",
    "load_dataset",
    "RagkitError",
    "ParseError",
    "SplitError",
    "EmbeddingError",
    "IndexingError",
    "RetrievalError",
    "GenerationError",
    "EvaluationError",
]
