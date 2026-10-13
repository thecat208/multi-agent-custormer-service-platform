"""
FAISS 本地向量知识库：构建与检索
构建：python -m retrieval.build（api 模式需 Embedding Key；local 模式需 sentence-transformers）
检索：search_kb(query, k) —— 未构建索引或嵌入不可用时返回空串，Agent 自动走关键词兜底
实现说明：不引入 langchain_community 的 VectorStore 封装，
直接用 faiss.IndexFlatIP + 归一化向量做余弦检索，元数据存 JSON，结构透明。
嵌入支持两种来源（EMBEDDING_PROVIDER）：
  api   —— OpenAI 兼容端点（默认，与 LLM 共用配置）
  local —— sentence-transformers 本地模型（可传本地目录），离线可用
"""

import json
import os
from typing import List

import numpy as np

from config import (
    EMBEDDING_DEVICE,
    EMBEDDING_LOCAL_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    KB_DOCS_DIR,
    KB_INDEX_PATH,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RETRIEVAL_TOP_K,
)

from log_setup import get_logger

logger = get_logger(__name__)

# 惰性加载的检索上下文（进程内单例）
_cache: dict = {}
# 嵌入实例单例：local 模式下避免每次查询重复加载模型
_embeddings_singleton: dict = {}


class _LocalEmbeddings:
    """sentence-transformers 本地模型的最小封装，接口对齐 OpenAIEmbeddings"""

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "本地嵌入模式需要 sentence-transformers：pip install sentence-transformers"
            ) from e
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            # 模型目录无效等情况：统一转为 RuntimeError，检索层据此降级
            raise RuntimeError(f"本地嵌入模型加载失败（{model_name}）: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs).tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vec).tolist()


def _embeddings_ready() -> bool:
    """嵌入是否已具备可用条件（不实际加载模型）"""
    if EMBEDDING_PROVIDER == "local":
        return bool(EMBEDDING_LOCAL_MODEL.strip())
    return bool(OPENAI_API_KEY)


def _get_embeddings():
    if EMBEDDING_PROVIDER == "local":
        if "local" not in _embeddings_singleton:
            _embeddings_singleton["local"] = _LocalEmbeddings(
                EMBEDDING_LOCAL_MODEL, EMBEDDING_DEVICE
            )
        return _embeddings_singleton["local"]

    if not OPENAI_API_KEY:
        return None
    if "api" not in _embeddings_singleton:
        from langchain_openai import OpenAIEmbeddings

        _embeddings_singleton["api"] = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            # 非 OpenAI 嵌入模型跳过 tiktoken 长度检查，原样发送字符串
            check_embedding_ctx_length=False,
            chunk_size=16,  # 兼容服务端批量上限
        )
    return _embeddings_singleton["api"]


def _iter_docs(docs_dir: str):
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(docs_dir, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        yield os.path.splitext(fname)[0], text


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-10, None)


def build_kb(docs_dir: str = None, index_path: str = None) -> str:
    """把 Markdown 文档切块、嵌入并写入 FAISS 索引 + JSON 元数据"""
    docs_dir = docs_dir or KB_DOCS_DIR
    index_path = index_path or KB_INDEX_PATH

    if not _embeddings_ready():
        raise RuntimeError(
            "嵌入不可用：api 模式需配置 OPENAI_API_KEY，"
            "local 模式需设置 EMBEDDING_LOCAL_MODEL"
        )
    embeddings = _get_embeddings()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )

    chunks, sources = [], []
    for title, text in _iter_docs(docs_dir):
        for piece in splitter.split_text(text):
            chunks.append(piece)
            sources.append(title)

    if not chunks:
        raise RuntimeError(f"{docs_dir} 下没有可索引的文档")

    vectors = _normalize(np.array(embeddings.embed_documents(chunks), dtype="float32"))

    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "sources": sources}, f, ensure_ascii=False)

    return f"indexed {len(chunks)} chunks from {len(set(sources))} docs -> {index_path}"


def _load(index_path: str):
    import faiss

    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"], meta["sources"]


def _get_context(index_path: str):
    key = os.path.abspath(index_path)
    if key not in _cache:
        if not (_embeddings_ready() and os.path.exists(os.path.join(index_path, "index.faiss"))):
            _cache[key] = None  # 未就绪：明确标记，避免每次请求重复探测
        else:
            try:
                _cache[key] = _load(index_path)
            except (ImportError, FileNotFoundError, RuntimeError) as e:
                # faiss 缺失或索引文件损坏：降级为不可用，而非让请求报错
                logger.warning(f"[vector_store] index load failed, degrading: {e}")
                _cache[key] = None
    return _cache[key]


def search_kb_with_sources(query: str, k: int = None,
                           index_path: str = None) -> List[tuple]:
    """检索知识库，返回 [(source, score, chunk)]；索引不可用时返回空列表"""
    context = _get_context(index_path or KB_INDEX_PATH)
    if not context or not query.strip():
        return []

    try:
        embeddings = _get_embeddings()
    except RuntimeError as e:
        # 本地模型缺失/依赖未装等：检索层降级为空结果，Agent 走关键词兜底
        logger.warning(f"[search_kb] embeddings unavailable: {e}")
        return []
    if embeddings is None:
        return []

    index, chunks, sources = context
    k = min(k or RETRIEVAL_TOP_K, len(chunks))

    qv = _normalize(np.array([embeddings.embed_query(query)], dtype="float32"))
    scores, ids = index.search(qv, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        results.append((sources[idx], float(score), chunks[idx]))
    return results


def search_kb(query: str, k: int = None, index_path: str = None) -> str:
    """检索知识库，返回拼接的参考片段；索引不可用时返回空串"""
    results = search_kb_with_sources(query, k, index_path)
    return "\n\n".join(
        f"【{source}】(相关度 {score:.2f})\n{chunk}" for source, score, chunk in results
    )
