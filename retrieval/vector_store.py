"""
FAISS 本地向量知识库：构建与检索
构建（需 Embedding API）：python -m retrieval.build
检索：search_kb(query, k) —— 未构建索引或未配置 Key 时返回空串，Agent 自动走关键词兜底
实现说明：不引入 langchain_community 的 VectorStore 封装，
直接用 faiss.IndexFlatIP + 归一化向量做余弦检索，元数据存 JSON，结构透明。
"""

import json
import os
from typing import List

import numpy as np

from config import (
    EMBEDDING_MODEL,
    KB_DOCS_DIR,
    KB_INDEX_PATH,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RETRIEVAL_TOP_K,
)

# 惰性加载的检索上下文（进程内单例）
_cache: dict = {}


def _get_embeddings():
    if not OPENAI_API_KEY:
        return None
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        # 非 OpenAI 嵌入模型跳过 tiktoken 长度检查，原样发送字符串
        check_embedding_ctx_length=False,
        chunk_size=16,  # 兼容服务端批量上限
    )


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

    embeddings = _get_embeddings()
    if embeddings is None:
        raise RuntimeError("未配置 OPENAI_API_KEY，无法构建知识库索引")

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
        if not (OPENAI_API_KEY and os.path.exists(os.path.join(index_path, "index.faiss"))):
            _cache[key] = None  # 未就绪：明确标记，避免每次请求重复探测
        else:
            _cache[key] = _load(index_path)
    return _cache[key]


def search_kb(query: str, k: int = None, index_path: str = None) -> str:
    """检索知识库，返回拼接的参考片段；索引不可用时返回空串"""
    context = _get_context(index_path or KB_INDEX_PATH)
    if not context or not query.strip():
        return ""

    index, chunks, sources = context
    k = min(k or RETRIEVAL_TOP_K, len(chunks))

    embeddings = _get_embeddings()
    if embeddings is None:
        return ""
    qv = _normalize(np.array([embeddings.embed_query(query)], dtype="float32"))
    scores, ids = index.search(qv, k)

    lines = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        lines.append(f"【{sources[idx]}】(相关度 {score:.2f})\n{chunks[idx]}")
    return "\n\n".join(lines)
