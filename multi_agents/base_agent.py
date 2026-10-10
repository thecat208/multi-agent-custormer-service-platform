"""
基础智能体类
所有专门智能体的基类：职责单一、只交换最小上下文。
对话历史统一从图状态 persisted_dialogue 读取（由 checkpointer 持久化，跨请求有效）。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

# 节点名 → 面向客户的中文名（转接提示与回复前缀使用）
AGENT_LABELS = {
    "product_agent": "产品专家",
    "tech_agent": "技术支持专家",
    "billing_agent": "账单专家",
    "complaint_agent": "投诉处理专家",
    "general_agent": "综合客服",
}


class HandoffDecision(BaseModel):
    """专家间转接判断的结构化输出"""
    handoff_to: str = Field(description="转接目标节点名；若无需转接填 none")
    reason: str = Field(default="", description="一句话转接理由")


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, expertise: List[str],
                 handoff_targets: Optional[Dict[str, str]] = None):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.llm = None  # 运行时注入
        # 可转接的专家：{节点名: 适用于该专家的诉求描述}
        self.handoff_targets = handoff_targets or {}

    def set_llm(self, llm):
        """设置 LLM 客户端"""
        self.llm = llm

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户查询"""
        ...

    def _decide_handoff(self, customer_query: str, state: Dict[str, Any]) -> str:
        """判断当前诉求是否应转接给更合适的专家。

        设计约束：
        - 只做单跳转接（handoff_count 上限 1），杜绝专家间循环转接
        - 模型输出的目标必须在 handoff_targets 白名单内，否则不转接
        - LLM 不可用或判断失败时返回空串，走本专家正常处理（宁可不转，不可中断）
        """
        if not self.handoff_targets or self.llm is None:
            return ""
        if state.get("handoff_count", 0) >= 1:
            return ""

        candidates = "\n".join(f"- {k}: {v}" for k, v in self.handoff_targets.items())
        prompt = f"""你是客服编排助手。判断当前客户诉求是否应该转接给更合适的专家处理。

可转接的专家：
{candidates}

若客户诉求明确属于上述某个专家的领域，输出对应节点名；否则输出 none。
只做判断，不要回答客户问题。"""
        try:
            result = self.llm.with_structured_output(HandoffDecision).invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=customer_query),
            ])
            target = (result.handoff_to or "").strip()
        except Exception as e:
            print(f"[{self.name}] handoff decision failed: {e}")
            return ""

        if target in self.handoff_targets:
            state["handoff_count"] = state.get("handoff_count", 0) + 1
            state["next_agent"] = target
            return target
        return ""

    def _retrieve(self, query: str, k: int = None) -> str:
        """知识库检索；未构建索引或未配置 Key 时返回空串（调用方走关键词兜底）"""
        try:
            from retrieval.vector_store import search_kb
            return search_kb(query, k)
        except Exception as e:
            print(f"[{self.name}] retrieval skipped: {e}")
            return ""

    def _run_with_tools(self, messages: List, tools: List, state: Dict[str, Any],
                        max_rounds: int = 3) -> str:
        """带工具调用的对话循环。

        工具执行异常以错误文本回传给 LLM 自行处理，不让异常冒泡中断整轮对话；
        max_rounds 限制防止模型循环调用。
        """
        tool_map = {t.name: t for t in tools}
        tool_llm = self.llm.bind_tools(tools)
        response = tool_llm.invoke(messages)

        for _ in range(max_rounds):
            calls = getattr(response, "tool_calls", None)
            if not calls:
                break
            messages.append(response)
            for call in calls:
                name = call["name"]
                tool = tool_map.get(name)
                if tool is None:
                    result = f"错误：未知工具 {name}"
                else:
                    try:
                        result = str(tool.invoke(call.get("args") or {}))
                    except Exception as e:
                        result = f"错误：工具执行失败 {e}"
                    state["tools_used"].append(name)
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            response = tool_llm.invoke(messages)

        content = response.content
        if isinstance(content, list):  # 部分模型返回分段内容
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return content

    def _get_conversation_context(self, state: Dict[str, Any], max_messages: int = 12) -> str:
        """读取最近对话并格式化为提示词上下文"""
        records = (state or {}).get("persisted_dialogue")
        if not records:
            return ""
        lines = []
        for msg in records[-max_messages:]:
            role = "用户" if msg.get("is_user", True) else "AI"
            lines.append(f"[{msg.get('timestamp', '')}] {role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _enhance_system_prompt_with_context(self, base_prompt: str) -> str:
        """增强系统提示，添加对话上下文说明"""
        context_instruction = """

重要：请结合对话历史上下文，理解客户之前的问题和需求，提供连贯、个性化的回答。
如果这是多轮对话，请参考之前的对话内容，避免重复信息，并基于客户的新问题提供补充信息。
保持对话的连贯性和自然性，让客户感受到你理解他们的完整需求。"""
        return base_prompt + context_instruction

    def get_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise,
        }
