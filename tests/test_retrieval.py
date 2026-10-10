"""检索层：索引未就绪时安全降级为空串（不抛异常、不调用 API）"""

import sys
import types

import numpy as np

import retrieval.vector_store as vs
from retrieval.vector_store import search_kb


def test_search_kb_degrades_to_empty_without_index():
    # 项目目录未构建索引（data/kb_index 不存在），必须返回空串而非报错
    assert search_kb("手机价格") == ""


def test_search_kb_empty_query():
    assert search_kb("   ") == ""


# ===== 嵌入双通道（api / local）=====


class _FakeSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([0.1, 0.2, 0.3])
        return np.tile(np.array([[0.1, 0.2, 0.3]]), (len(texts), 1))


def _fresh(monkeypatch, provider, **attrs):
    """重置单例并切换 provider，返回打补丁后的模块"""
    monkeypatch.setattr(vs, "_embeddings_singleton", {})
    monkeypatch.setattr(vs, "EMBEDDING_PROVIDER", provider)
    for k, v in attrs.items():
        monkeypatch.setattr(vs, k, v)
    return vs


def test_local_embeddings_factory(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    mod = _fresh(monkeypatch, "local", EMBEDDING_LOCAL_MODEL="D:/models/bge")

    emb = mod._get_embeddings()
    assert emb is not None
    # 单例：重复获取不重复加载模型
    assert mod._get_embeddings() is emb
    # 接口对齐 OpenAIEmbeddings
    doc_vecs = emb.embed_documents(["a", "b"])
    assert len(doc_vecs) == 2 and len(doc_vecs[0]) == 3
    assert len(emb.embed_query("q")) == 3
    assert mod._embeddings_ready() is True


def test_local_mode_without_dependency_raises_runtime_error(monkeypatch):
    # sys.modules 置 None 可让 import 语句抛 ImportError
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    mod = _fresh(monkeypatch, "local", EMBEDDING_LOCAL_MODEL="bge")

    try:
        mod._get_embeddings()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "sentence-transformers" in str(e)


def test_api_mode_without_key_returns_none(monkeypatch):
    mod = _fresh(monkeypatch, "api", OPENAI_API_KEY="")
    assert mod._get_embeddings() is None
    assert mod._embeddings_ready() is False


def test_local_mode_with_blank_model_not_ready(monkeypatch):
    mod = _fresh(monkeypatch, "local", EMBEDDING_LOCAL_MODEL="   ")
    assert mod._embeddings_ready() is False
