"""日志配置行为验证（离线）"""

import logging

from log_setup import setup_logging


def test_setup_logging_is_idempotent():
    root = logging.getLogger()
    before = len(root.handlers)
    setup_logging()
    setup_logging()
    assert len(root.handlers) == before  # 重复调用不叠加 handler


def test_third_party_info_noise_silenced():
    setup_logging()
    # faiss 加载信息等第三方 INFO 日志不应再冒到 root handler
    assert logging.getLogger("faiss.loader").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
    # 自家模块不受影响（未设置级别的 logger 返回 NOTSET，继承 root 的 INFO）
    assert logging.getLogger("multi_agents.base_agent").level == logging.NOTSET
