"""评测指标计算的纯函数部分：与 IO / LLM 解耦，便于离线测试"""

from typing import Any, Dict, List, Optional, Tuple


def rank_of_expected(sources: List[str], expected: str) -> Optional[int]:
    """期望文档在检索结果中的名次（1-based）；未命中返回 None"""
    for i, source in enumerate(sources, start=1):
        if source == expected:
            return i
    return None


def judge_tool_case(used_tools: List[str], recorded_args: Dict[str, List[dict]],
                    case: Dict[str, Any]) -> Tuple[bool, Optional[bool]]:
    """判定单条工具用例。

    返回 (tool_ok, arg_ok)：
    - tool_ok：required_tools 全部被调用，且 forbidden_tools 均未被调用
    - arg_ok：用例带 expected_args 时为 bool（该工具最近一次调用参数与期望一致）；
      用例不带 expected_args 时为 None（不参与参数准确率统计）
    """
    used = list(used_tools or [])
    tool_ok = (
        all(t in used for t in case.get("required_tools", []))
        and not any(t in used for t in case.get("forbidden_tools", []))
    )

    expected_args = case.get("expected_args") or {}
    if not expected_args:
        return tool_ok, None

    for tool_name, expect in expected_args.items():
        calls = recorded_args.get(tool_name) or []
        if not calls:
            return tool_ok, False
        last = calls[-1]
        for key, want in expect.items():
            got = str(last.get(key, "") or "").strip().upper()
            if got != str(want).strip().upper():
                return tool_ok, False
    return tool_ok, True


def aggregate_faithfulness(claims: List[Dict[str, Any]]) -> Optional[float]:
    """单条回答的忠实度 = 被支持声明占比；无事实性声明时返回 None"""
    factual = [c for c in claims if isinstance(c, dict) and "supported" in c]
    if not factual:
        return None
    return sum(1 for c in factual if c["supported"]) / len(factual)


def compute_cost(input_tokens: float, output_tokens: float,
                 price_input_per_m: float, price_output_per_m: float) -> float:
    """按单价（元 / 百万 token）计算成本（元）"""
    return (input_tokens / 1e6 * price_input_per_m
            + output_tokens / 1e6 * price_output_per_m)


def judge_task_case(response: str, tools_used: List[str],
                    criteria: Dict[str, Any]) -> bool:
    """判定单条端到端用例是否完成任务。

    criteria（均可选）：
    - keywords_all: 回答必须包含的全部关键词
    - any_groups:   每组关键词至少命中一个（组与组之间是 AND）
    - tools_any:    至少调用了其中一个工具
    判定为大小写不敏感的子串匹配；空回答直接判失败。
    """
    if not (response or "").strip():
        return False
    haystack = response.lower()

    for kw in criteria.get("keywords_all", []):
        if kw.lower() not in haystack:
            return False
    for group in criteria.get("any_groups", []):
        if group and not any(kw.lower() in haystack for kw in group):
            return False

    tools_any = criteria.get("tools_any") or []
    if tools_any and not any(t in (tools_used or []) for t in tools_any):
        return False
    return True
