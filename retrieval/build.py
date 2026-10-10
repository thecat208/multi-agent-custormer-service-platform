"""
知识库构建 CLI：python -m retrieval.build
嵌入来源由 EMBEDDING_PROVIDER 决定：
  api   —— .env 配置 OPENAI_API_KEY（默认）
  local —— pip install sentence-transformers，EMBEDDING_LOCAL_MODEL 指向模型名或本地目录
"""

from retrieval.vector_store import build_kb

if __name__ == "__main__":
    print(build_kb())
