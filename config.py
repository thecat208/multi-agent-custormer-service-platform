"""
配置文件
所有参数均可通过 .env 环境变量覆盖
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ===== LLM（OpenAI 兼容端点，默认硅基流动）=====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
# 硅基流动的 Qwen3 系列默认开启思考模式，客服场景关闭以降低延迟；
# 切换到其他供应商时可设 LLM_ENABLE_THINKING=true 兼容
LLM_ENABLE_THINKING = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"

# ===== Embedding（知识库检索用）=====
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# ===== 会话持久化（SQLite checkpointer）=====
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "./data/checkpoints.sqlite")

# ===== Web 服务 =====
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "5000"))

# ===== 系统 =====
SYSTEM_NAME = "智能客服工作台"
VERSION = "2.0.0"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
