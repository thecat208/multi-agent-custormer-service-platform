"""评测脚本：用例加载与指标统计（不需要 API Key）"""

import evaluation.run_eval as ev


def test_golden_set_loads_30_cases():
    cases = ev.load_cases()
    assert len(cases) == 30
    for c in cases:
        assert c["query"].strip()
        assert c["query_type"] in {
            "product_info", "technical_support", "billing",
            "complaint", "general_inquiry", "out_of_scope",
        }


def test_summarize_metrics():
    fake = [
        {"passed": True, "expected": "billing", "latency_ms": 100},
        {"passed": False, "expected": "billing", "latency_ms": 200},
        {"passed": True, "expected": "out_of_scope", "latency_ms": 300},
    ]
    m = ev.summarize(fake)
    assert m["total"] == 3
    assert m["passed"] == 2
    assert abs(m["accuracy"] - 0.667) < 0.01
    assert m["refusal_recall"] == 1.0
    assert m["per_class"]["billing"]["accuracy"] == 0.5
    assert m["latency_p95_ms"] >= m["latency_avg_ms"]
