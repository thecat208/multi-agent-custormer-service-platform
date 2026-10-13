"""
多智能体客服系统 - LangGraph 工作流
意图分类（含 out_of_scope 护栏）→ 专家路由 → 汇总响应
对话轮次持久化在图状态 persisted_dialogue 中，由 checkpointer 落盘，跨请求可续聊
"""

import uuid
from typing import Dict, List, Any, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph

from config import *
from llm import get_llm
from multi_agents import (
    ProductAgent, TechAgent, BillingAgent,
    ComplaintAgent, GeneralAgent
)
from tools import classify_query

from log_setup import get_logger

logger = get_logger(__name__)

# 超出客服范围时的固定回复（护栏：不调用业务智能体）
OUT_OF_SCOPE_REPLY = (
    "抱歉，这里是智能客服，仅处理与产品、技术、账单、投诉及相关售后政策类问题；"
    "请用一句话说明您的具体业务诉求，我很乐意协助。"
)


class AgentState(TypedDict):
    session_id: str
    messages: List[Any]
    current_agent: str
    customer_query: str
    query_type: str
    response: str
    tools_used: List[str]
    # 专家间转接：目标节点名（空串表示无转接）与已发生的转接次数（上限 1，防循环）
    next_agent: str
    handoff_count: int
    # 由图 checkpointer 持久化，跨请求续聊的对话记录
    persisted_dialogue: List[Any]


def initialize_agents():
    """初始化所有智能体，注入 LLM 与会话管理器"""
    agents = {
        "product_agent": ProductAgent(),
        "tech_agent": TechAgent(),
        "billing_agent": BillingAgent(),
        "complaint_agent": ComplaintAgent(),
        "general_agent": GeneralAgent()
    }

    for agent in agents.values():
        agent.set_llm(get_llm())

    return agents


def classify_query_node(state: AgentState) -> AgentState:
    """意图分类 + out_of_scope 护栏"""
    try:
        from langgraph.config import get_config
        cfg = get_config()
        tid = (cfg.get("configurable") or {}).get("thread_id")
        if tid:
            state["session_id"] = str(tid)
    except RuntimeError:
        pass

    # 首次进入时初始化各状态字段
    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())
    state.setdefault("tools_used", [])
    state.setdefault("persisted_dialogue", [])
    state.setdefault("next_agent", "")
    state.setdefault("handoff_count", 0)
    state.setdefault("messages", [])

    customer_query = state.get("customer_query", "")

    if not customer_query:
        # 空输入与非客服请求共用同一条短路路径：固定回复，不进入业务智能体
        state["response"] = OUT_OF_SCOPE_REPLY
        state["query_type"] = "out_of_scope"
        state["current_agent"] = "智能客服"
        state["tools_used"].append("empty_query_refusal")
        return state

    try:
        state["query_type"] = classify_query(customer_query, get_llm())
    except Exception as e:
        logger.warning(f"[classify_query_node] 意图分类失败，降级 general_inquiry: {e}")
        state["query_type"] = "general_inquiry"

    state["tools_used"].append("query_classification")

    # 用户轮次写入 checkpointer 持久化的对话
    pd = list(state.get("persisted_dialogue") or [])
    pd.append({
        "content": str(customer_query),
        "is_user": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    state["persisted_dialogue"] = pd

    # 护栏：超出范围直接固定回复，不进入业务智能体
    if state["query_type"] == "out_of_scope":
        state["response"] = OUT_OF_SCOPE_REPLY
        state["current_agent"] = "智能客服"
        pd.append({
            "content": OUT_OF_SCOPE_REPLY,
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["persisted_dialogue"] = pd
        state["tools_used"].append("out_of_scope_refusal")

    return state


def create_agent_node(agent_name: str):
    """创建智能体处理节点"""
    def agent_node(state: AgentState) -> AgentState:
        # 清空上一跳的转接标记，避免路由读到陈旧目标造成循环
        state["next_agent"] = ""

        agents = initialize_agents()
        agent = agents.get(agent_name)
        if not agent:
            state["response"] = f"Error: Agent {agent_name} not found"
            return state

        result = agent.process(state)

        if not isinstance(result, dict):
            logger.error(f"Agent {agent_name} returned non-dict result: {type(result)}")
            result = {"response": "Error: Agent processing failed", "current_agent": agent_name}

        if "response" not in result:
            logger.error(f"Agent {agent_name} result missing 'response' field: {result}")
            result["response"] = "Error: No response from agent"

        # 转接轮：本节点未产生回复，对话轮次由最终应答的专家写入
        if not result.get("response"):
            return result

        # 助手轮次写入 checkpointer 状态
        pd = list(result.get("persisted_dialogue") or [])
        pd.append({
            "content": str(result["response"]),
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        result["persisted_dialogue"] = pd

        return result
    return agent_node


def final_response_node(state: AgentState) -> AgentState:
    """汇总响应：标注当前处理专家"""
    state["response"] = f"【{state['current_agent']}】\n{state['response']}"
    return state


def make_graph(checkpointer=None):
    """构建 LangGraph 工作流图（可选注入 checkpointer 用于跨请求持久化）"""
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("product_agent", create_agent_node("product_agent"))
    workflow.add_node("tech_agent", create_agent_node("tech_agent"))
    workflow.add_node("billing_agent", create_agent_node("billing_agent"))
    workflow.add_node("complaint_agent", create_agent_node("complaint_agent"))
    workflow.add_node("general_agent", create_agent_node("general_agent"))
    workflow.add_node("final_response", final_response_node)

    workflow.set_entry_point("classify_query")

    workflow.add_conditional_edges(
        "classify_query",
        lambda x: x.get("query_type", ""),
        {
            "product_info": "product_agent",
            "technical_support": "tech_agent",
            "billing": "billing_agent",
            "complaint": "complaint_agent",
            "general_inquiry": "general_agent",
            "out_of_scope": "final_response",
        }
    )

    # 专家节点之后：若有受控转接（next_agent）则去目标专家，否则进最终响应。
    # 转接目标受 handoff_targets 白名单与 handoff_count 上限双重约束，不会成环。
    def route_after_agent(x):
        return x.get("next_agent") or "final_response"

    agent_names = ("product_agent", "tech_agent", "billing_agent",
                   "complaint_agent", "general_agent")
    for name in agent_names:
        targets = {a: a for a in agent_names if a != name}
        targets["final_response"] = "final_response"
        workflow.add_conditional_edges(name, route_after_agent, targets)

    workflow.set_finish_point("final_response")

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("[graph] customer service workflow compiled")
    return app


if __name__ == "__main__":
    make_graph()
    logger.info("multi-agent customer service graph built successfully")
