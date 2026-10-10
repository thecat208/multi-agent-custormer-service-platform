"""
知识库构建 CLI：python -m retrieval.build
需在 .env 配置 OPENAI_API_KEY（Embedding 走同一 OpenAI 兼容端点）。
"""

from retrieval.vector_store import build_kb

if __name__ == "__main__":
    print(build_kb())
