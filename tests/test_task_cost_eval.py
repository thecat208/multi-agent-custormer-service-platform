"""任务完成率判定与成本核算的纯函数 / 回调逻辑（离线）"""

import json
import os

from evaluation.golden_utils import compute_cost, judge_task_case
from evaluation.run_cost_eval import TokenUsageHandler
from evaluation.run_metrics_eval import _load_jsonl
from evaluation.run_task_eval import _load_jsonl as _load_task_jsonl

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evaluation")


def test_judge_task_case_keywords():
    criteria = {"any_groups": [["7天", "七天"], ["未激活", "未使用"]]}
    resp = "七天无理由退货要求商品未激活、未使用，包装完整。"
    assert judge_task_case(resp, [], criteria)
    # 缺一个关键词组
    assert not judge_task_case("支持七天无理由退货。", [], criteria)
    # 空回答直接失败
    assert not judge_task_case("", [], criteria)


def test_judge_task_case_tools_and_all():
    criteria = {"tools_any": ["query_order"], "keywords_all": ["已发货"]}
    assert judge_task_case("订单已发货，运输中。", ["query_order", "kb_retrieval"], criteria)
    # 缺工具
    assert not judge_task_case("订单已发货。", [], criteria)
    # 有关键词外的要求未满足
    assert not judge_task_case("订单已签收。", ["query_order"], criteria)


def test_compute_cost():
    # 1000 输入 + 500 输出，单价 2/8 元每百万
    cost = compute_cost(1000, 500, 2.0, 8.0)
    assert abs(cost - (1000 / 1e6 * 2 + 500 / 1e6 * 8)) < 1e-9
    assert compute_cost(0, 0, 2.0, 8.0) == 0


class _FakeResponse:
    """模拟 ChatCompletionResult：llm_output 携带 token_usage"""

    def __init__(self, prompt_tokens, completion_tokens):
        self.llm_output = {"token_usage": {"prompt_tokens": prompt_tokens,
                                           "completion_tokens": completion_tokens}}


def test_token_usage_handler_aggregates():
    handler = TokenUsageHandler()
    handler.on_llm_end(_FakeResponse(100, 50))
    handler.on_llm_end(_FakeResponse(200, 30))
    handler.on_llm_end(_FakeResponse(0, 0))  # 空用量不贡献也不报错
    assert handler.input_tokens == 300
    assert handler.output_tokens == 80
    assert handler.calls == 3
    assert handler.total_tokens == 380


def test_golden_files_valid():
    # 任务 golden：criteria 字段合法、query 非空
    valid_tools = {"query_order", "create_ticket"}
    cases = _load_task_jsonl("task_golden.jsonl")
    assert len(cases) >= 10
    for c in cases:
        assert c["query"].strip()
        criteria = c["criteria"]
        tools = criteria.get("tools_any", [])
        assert all(t in valid_tools for t in tools)
        for group in criteria.get("any_groups", []):
            assert group, "any_groups 中的组不能为空"
    # 工具 golden（既有文件，防回归）
    for c in _load_jsonl("tool_golden.jsonl"):
        assert c["agent"] in ("billing_agent", "tech_agent")
