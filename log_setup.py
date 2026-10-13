"""
统一日志配置
模块内通过 get_logger(__name__) 获取 logger；入口（app.py / CLI）无需显式
初始化——首次 get_logger 时自动按 LOG_LEVEL 完成根 logger 配置。
诊断信息分级：warning = 已自动降级/兜底；error = 请求内失败但已返回兜底回复。
"""

import logging
import sys

from config import LOG_LEVEL

_CONFIGURED = False
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str = None) -> None:
    """配置根 logger（幂等，可安全多次调用）"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.setLevel((level or LOG_LEVEL).upper())
    root.addHandler(handler)

    # 第三方库的 INFO 噪音（faiss 加载过程、HTTP 客户端等）只保留 WARNING 以上
    for name in ("faiss", "faiss.loader", "urllib3", "httpx", "httpx2", "httpcore",
                 "openai", "langchain", "langchain_core", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # uvicorn 自带 handler，关闭向 root 传播避免重复输出
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
