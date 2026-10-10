"""工具调用循环：用脚本化的假 LLM 离线驱动，验证真实工具被执行且错误可回传"""

from langchain_core.messages import AIMessage

from multi_agents.billing_agent import BillingAgent
from tools.query_tools import QueryClassification, normalize_classifier_label  # noqa: F401
from tools.business import query_order


def _ai(text="", tool_calls=None):
    return AIMessage(content=text, tool_calls=tool_calls or [])


class _ScriptedLLM:
    """按脚本依次返回 AIMessage；记录每次收到的消息序列供断言"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.script.pop(0)


def _new_state(query):
    return {
        "customer_query": query,
        "persisted_dialogue": [],
        "tools_used": [],
    }


def test_tool_call_happy_path():
    agent = BillingAgent()
    agent.set_llm(_ScriptedLLM([
        _ai(tool_calls=[{"name": "query_order", "args": {"order_id": "ORD20241001001"}, "id": "call1"}]),
        _ai("您的订单 X1 Pro 已发货，顺丰运输中，预计明天送达。"),
    ]))
    state = _new_state("帮我查下订单 ORD20241001001")

    result = agent.process(state)

    assert "已发货" in result["response"]
    assert "query_order" in result["tools_used"]
    assert result["current_agent"] == "账单专家"


def test_tool_error_returned_to_llm():
    """工具报错时错误文本回传给 LLM，不中断流程"""
    llm = _ScriptedLLM([
        _ai(tool_calls=[{"name": "query_order", "args": {"order_id": "ORD_NOT_EXIST"}, "id": "call1"}]),
        _ai("没有查到这个订单，请您核对订单号后再试。"),
    ])
    agent = BillingAgent()
    agent.set_llm(llm)
    state = _new_state("查订单")

    result = agent.process(state)

    assert "核对订单号" in result["response"]
    assert "query_order" in result["tools_used"]
    # 第二轮对话里应包含工具返回的错误信息（ToolMessage）
    second_round = llm.calls[1]
    assert any(getattr(m, "tool_call_id", None) == "call1" for m in second_round)


def test_unknown_tool_name_is_handled():
    llm = _ScriptedLLM([
        _ai(tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "call1"}]),
        _ai("好的，我换一种方式为您解答。"),
    ])
    agent = BillingAgent()
    agent.set_llm(llm)
    state = _new_state("随便问问")

    result = agent.process(state)

    assert result["response"] == "好的，我换一种方式为您解答。"
    # 未注册的工具不计入 tools_used
    assert "nonexistent_tool" not in result["tools_used"]


def test_no_llm_degrades_gracefully():
    agent = BillingAgent()
    agent.set_llm(None)
    state = _new_state("退款政策")

    result = agent.process(state)

    assert "系统错误" in result["response"]
    assert result["current_agent"] == "账单专家"
