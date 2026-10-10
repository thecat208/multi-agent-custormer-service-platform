"""专家间受控转接：白名单单跳 + 循环防护 + 转接轮不写对话"""

import multi_agent_customer_service as mas
from multi_agent_customer_service import make_graph
from multi_agents.billing_agent import BillingAgent
from multi_agents.general_agent import GeneralAgent
from multi_agents.product_agent import ProductAgent


def _patch_decide(agent_cls, target):
    import types
    def decide(self, customer_query, state):
        if state.get("handoff_count", 0) >= 1 or not target:
            return ""
        state["handoff_count"] = state.get("handoff_count", 0) + 1
        state["next_agent"] = target
        return target
    agent_cls._decide_handoff = decide


def test_handoff_general_to_billing(monkeypatch):
    """综合客服把账单诉求转给账单专家，最终回复只由账单专家产生"""
    monkeypatch.setattr(mas, "classify_query", lambda q, llm=None: "general_inquiry")
    _patch_decide(GeneralAgent, "billing_agent")

    g = make_graph()
    result = g.invoke({"customer_query": "我要退款", "session_id": "ho1"})

    assert result["current_agent"] == "账单专家"
    assert "handoff:billing_agent" in result["tools_used"]
    # 转接轮不写对话：只有用户一轮 + 账单专家回复一轮
    assert len(result["persisted_dialogue"]) == 2
    assert result["persisted_dialogue"][1]["is_user"] is False


def test_handoff_loop_guard(monkeypatch):
    """转接上限 1 跳：即使多个专家都要求继续转接，也会被 handoff_count 拦住"""
    monkeypatch.setattr(mas, "classify_query", lambda q, llm=None: "general_inquiry")
    _patch_decide(GeneralAgent, "billing_agent")
    _patch_decide(BillingAgent, "product_agent")
    _patch_decide(ProductAgent, "billing_agent")

    g = make_graph()
    result = g.invoke({"customer_query": "随便说说", "session_id": "ho2"})

    # general → billing（第 1 跳），billing 的再转接被上限拦下，回复由 billing 产生
    assert result["current_agent"] == "账单专家"
    assert "handoff:product_agent" not in result["tools_used"]
    assert len(result["persisted_dialogue"]) == 2


def test_no_handoff_when_not_triggered(monkeypatch):
    """判断为不转接时走正常应答路径"""
    monkeypatch.setattr(mas, "classify_query", lambda q, llm=None: "general_inquiry")
    _patch_decide(GeneralAgent, "")

    g = make_graph()
    result = g.invoke({"customer_query": "营业时间", "session_id": "ho3"})

    assert result["current_agent"] == "综合客服"
    assert not any(str(t).startswith("handoff:") for t in result["tools_used"])
