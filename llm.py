"""
LLM 客户端工厂
统一通过 langchain-openai 的 ChatOpenAI 访问 OpenAI 兼容端点（默认硅基流动），
原生获得流式输出、function calling 与结构化输出能力，不再手写 HTTP 客户端。
"""

from functools import lru_cache

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
    LLM_ENABLE_THINKING,
)

from log_setup import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    """获取全局 LLM 实例。

    未配置 API Key 时返回 None 而不是抛异常：
    图仍可编译、路由与护栏逻辑可离线测试，LLM 依赖的节点走各自的异常分支。
    """
    if not OPENAI_API_KEY:
        logger.warning("[llm] OPENAI_API_KEY not configured, LLM features degraded")
        return None

    from langchain_openai import ChatOpenAI

    # 硅基流动 Qwen3 系列默认开启思考模式，客服场景关闭以降低首 token 延迟
    extra_body = None if LLM_ENABLE_THINKING else {"enable_thinking": False}

    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        extra_body=extra_body,
    )
