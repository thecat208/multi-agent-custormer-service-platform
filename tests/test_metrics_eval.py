"""评测指标纯函数与 golden 文件校验（全部离线可跑）"""

import json
import os

from evaluation.golden_utils import (
    aggregate_faithfulness,
    judge_tool_case,
    rank_of_expected,
)
from evaluation.run_metrics_eval import _load_jsonl

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evaluation")


def test_rank_of_expected():
    assert rank_of_expected(["a", "b", "c"], "a") == 1
    assert rank_of_expected(["a", "b", "c"], "c") == 3
    assert rank_of_expected(["a", "b"], "z") is None
    assert rank_of_expected([], "a") is None


def test_judge_tool_case_requires_and_forbids():
    case = {"required_tools": ["query_order"], "forbidden_tools": ["create_ticket"]}
    ok, arg = judge_tool_case(["query_order"], {}, case)
    assert ok and arg is None
    # 缺必需工具
    ok, _ = judge_tool_case([], {}, case)
    assert not ok
    # 出现禁止工具
    ok, _ = judge_tool_case(["query_order", "create_ticket"], {}, case)
    assert not ok


def test_judge_tool_case_args():
    case = {"required_tools": ["query_order"], "forbidden_tools": [],
            "expected_args": {"query_order": {"order_id": "ORD20241001001"}}}
    recorded = {"query_order": [{"order_id": "ord20241001001"}]}  # 大小写差异应视为一致
    ok, arg_ok = judge_tool_case(["query_order"], recorded, case)
    assert ok and arg_ok is True

    # 参数错误
    recorded_bad = {"query_order": [{"order_id": "ORD00000000000"}]}
    _, arg_ok = judge_tool_case(["query_order"], recorded_bad, case)
    assert arg_ok is False

    # 工具没被调用（有参数期望即判失败）
    _, arg_ok = judge_tool_case(["query_order"], {}, case)
    assert arg_ok is False

    # 多次调用取最近一次
    recorded_multi = {"query_order": [{"order_id": "ORD00000000000"},
                                      {"order_id": "ORD20241001001"}]}
    _, arg_ok = judge_tool_case(["query_order"], recorded_multi, case)
    assert arg_ok is True


def test_aggregate_faithfulness():
    claims = [{"claim": "a", "supported": True},
              {"claim": "b", "supported": True},
              {"claim": "c", "supported": False}]
    assert aggregate_faithfulness(claims) == 2 / 3
    # 无事实性声明 → None（不参与统计）
    assert aggregate_faithfulness([]) is None
    assert aggregate_faithfulness([{"claim": "问候语"}]) is None


def test_golden_files_valid():
    # 检索 golden：query 非空、expected_doc 在已知知识库范围内
    valid_docs = {"product_phone", "product_laptop", "product_audio",
                  "policy_refund", "policy_payment", "policy_logistics"}
    cases = _load_jsonl("retrieval_golden.jsonl")
    assert len(cases) >= 10
    for c in cases:
        assert c["query"].strip()
        assert c["expected_doc"] in valid_docs

    # 工具 golden：required/forbidden 不相交，agent 合法
    for c in _load_jsonl("tool_golden.jsonl"):
        assert c["agent"] in ("billing_agent", "tech_agent")
        assert not set(c["required_tools"]) & set(c["forbidden_tools"])
        assert all(t in ("query_order", "create_ticket")
                   for t in c["required_tools"] + c["forbidden_tools"])


def test_load_jsonl_ignores_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert len(rows) == 2


def test_parse_claims_text_fallback():
    from evaluation.run_metrics_eval import parse_claims_text

    # 正常 JSON 数组（可能夹杂说明文字）
    text = '核验结果如下：[{"claim": "4999元", "supported": true}, {"claim": "90天退货", "supported": false}]'
    claims = parse_claims_text(text)
    assert len(claims) == 2
    assert claims[0] == {"claim": "4999元", "supported": True}
    assert claims[1]["supported"] is False

    # 无 JSON 数组 → 空列表（该题记为未核验，不硬崩）
    assert parse_claims_text("无法解析") == []

    # 多行 JSON
    text_ml = '结果：\n[\n  {"claim": "支持支付宝", "supported": true}\n]'
    assert parse_claims_text(text_ml)[0]["claim"] == "支持支付宝"
