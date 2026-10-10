"""
基础智能体类
所有专门智能体的基类：职责单一、只交换最小上下文。
对话历史统一从图状态 persisted_dialogue 读取（由 checkpointer 持久化，跨请求有效）。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain_core.messages import ToolMessage


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, expertise: List[str]):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.llm = None  # 运行时注入

    def set_llm(self, llm):
        """设置 LLM 客户端"""
        self.llm = llm

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户查询"""
        ...

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
