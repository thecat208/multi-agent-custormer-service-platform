"""
异常注入场景集：每类故障注入后断言系统按设计降级或自愈。
全部离线可跑（不依赖 API Key / faiss / 本地嵌入模型）。
每个场景返回 (名称, 是否恢复, 说明)；异常恢复率 = 恢复场景数 / 总场景数。
"""

import json
from typing import Callable, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage

GENERAL_FALLBACK_MARK = "general_inquiry"


def scenario_classify_llm_failure() -> Tuple[str, bool, str]:
    """分类 LLM 异常 → 降级 general_inquiry，图正常完成"""
    import multi_agent_customer_service as mas

    original = mas.classify_query

    def _boom(query, llm=None):
        raise RuntimeError("simulated LLM timeout")

    mas.classify_query = _boom
    try:
        graph = mas.make_graph()
        result = graph.invoke({"customer_query": "X1 Pro 多少钱"})
    finally:
        mas.classify_query = original

    ok = bool(result.get("response")) and result.get("query_type") == GENERAL_FALLBACK_MARK
    return "分类 LLM 异常", ok, f"query_type={result.get('query_type')}，回复非空={bool(result.get('response'))}"


def scenario_tool_invalid_args() -> Tuple[str, bool, str]:
    """工具收到无效参数 → 返回结构化错误，不抛异常"""
    from tools.business import create_ticket, query_order

    r1 = json.loads(query_order.invoke({"order_id": ""}))
    r2 = json.loads(create_ticket.invoke({"device": "", "symptom": ""}))
    ok = "error" in r1 and "error" in r2
    return "工具无效参数", ok, "两个工具均返回结构化错误提示"


def scenario_index_missing() -> Tuple[str, bool, str]:
    """索引缺失 → 检索降级空串（Agent 走关键词兜底）"""
    from retrieval.vector_store import search_kb

    result = search_kb("手机价格", index_path="./no/such/index")
    ok = result == ""
    return "索引缺失", ok, "search_kb 返回空串而非异常"


def scenario_embedding_unavailable() -> Tuple[str, bool, str]:
    """本地嵌入模型不可用 → 检索降级空串

    用一个存在但不是合法模型目录的路径触发加载失败（避免依赖"未安装 st"
    或"路径不存在会触发 HF 下载"这两种环境差异）。
    """
    import tempfile

    import retrieval.vector_store as vs

    bogus_dir = tempfile.mkdtemp(prefix="not_a_model_")  # 存在但无 config.json
    old_provider, old_model = vs.EMBEDDING_PROVIDER, vs.EMBEDDING_LOCAL_MODEL
    vs.EMBEDDING_PROVIDER, vs.EMBEDDING_LOCAL_MODEL = "local", bogus_dir
    try:
        result = vs.search_kb("手机价格")
    finally:
        vs.EMBEDDING_PROVIDER, vs.EMBEDDING_LOCAL_MODEL = old_provider, old_model
    ok = result == ""
    return "嵌入模型不可用", ok, "search_kb 降级为空串"


def scenario_unknown_handoff_target() -> Tuple[str, bool, str]:
    """转接目标不在白名单 → 拒绝转接，专家正常应答"""
    from multi_agents import GeneralAgent
    from multi_agents.base_agent import HandoffDecision

    class _FakeStructuredLLM:
        def __init__(self, decision):
            self._decision = decision

        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return self._decision

    agent = GeneralAgent()
    agent.llm = _FakeStructuredLLM(HandoffDecision(handoff_to="ceo_agent", reason="越权转接"))
    state = {"handoff_count": 0, "next_agent": ""}
    target = agent._decide_handoff("你们老板是谁", state)
    ok = target == "" and state["handoff_count"] == 0 and state["next_agent"] == ""
    return "转接目标越权", ok, "白名单外目标被拒绝，未发生转接"


class _AlwaysToolCallLLM:
    """每次 invoke 都请求调用工具，用于验证有界循环封顶"""

    def __init__(self):
        self._n = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._n += 1
        return AIMessage(
            content="",
            tool_calls=[{"name": "query_order", "args": {"order_id": "ORD20241001001"},
                         "id": f"call_{self._n}"}],
        )


def scenario_tool_loop_bounded() -> Tuple[str, bool, str]:
    """LLM 无限请求工具 → 有界循环 3 轮封顶，不无限循环"""
    from multi_agents import BillingAgent
    from tools.business import query_order

    agent = BillingAgent()
    agent.llm = _AlwaysToolCallLLM()
    state = {"tools_used": []}
    content = agent._run_with_tools(
        [HumanMessage(content="查订单 ORD20241001001")], [query_order], state, max_rounds=3
    )
    calls = state["tools_used"].count("query_order")
    ok = calls <= 3
    return "工具循环失控", ok, f"循环 {calls} 轮后被封顶（上限 3）"


def scenario_llm_unavailable() -> Tuple[str, bool, str]:
    """LLM 完全不可用 → 兜底话术，不崩溃"""
    from multi_agents import BillingAgent

    agent = BillingAgent()
    agent.llm = None
    result = agent.process({
        "customer_query": "查一下订单 ORD20241001001",
        "tools_used": [],
        "persisted_dialogue": [],
        "session_id": "resilience-llm-none",
    })
    ok = bool(result.get("response"))
    return "LLM 不可用", ok, f"返回兜底回复：{result.get('response', '')[:30]}"


SCENARIOS: List[Callable[[], Tuple[str, bool, str]]] = [
    scenario_classify_llm_failure,
    scenario_tool_invalid_args,
    scenario_index_missing,
    scenario_embedding_unavailable,
    scenario_unknown_handoff_target,
    scenario_tool_loop_bounded,
    scenario_llm_unavailable,
]
