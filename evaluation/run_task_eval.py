"""
任务完成率评测：端到端跑完整图，按 golden 中的成功标准判定每例是否完成用户诉求。

用法：python -m evaluation.run_task_eval（需 API Key + 已建索引）
产物：evaluation/task_report.md
判定标准（task_golden.jsonl 的 criteria 字段）：
- any_groups：每组关键词至少命中一个（组间 AND）
- tools_any：至少调用了其中一个业务工具
- 关键词为大小写不敏感子串匹配；空回答直接判失败
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List

from evaluation.golden_utils import judge_task_case
from evaluation.run_metrics_eval import _load_jsonl
from log_setup import get_logger

logger = get_logger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_REPORT_PATH = os.path.join(EVAL_DIR, "task_report.md")


def eval_task() -> Dict[str, Any]:
    from app import get_graph

    cases = _load_jsonl("task_golden.jsonl")
    graph = get_graph()
    # 会话 id 按次隔离：避免 checkpointer 历史跨评测轮次累积，污染多轮判定
    run_id = str(int(time.time()))
    passed, latencies, failures = 0, [], []

    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        final = graph.invoke(
            {"customer_query": case["query"], "session_id": f"eval-task-{run_id}-{i}"},
            {"configurable": {"thread_id": f"eval-task-{run_id}-{i}"}},
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        response = final.get("response", "")
        tools_used = [t for t in final.get("tools_used", [])]
        ok = judge_task_case(response, tools_used, case.get("criteria", {}))
        passed += ok
        status = "OK " if ok else "FAIL"
        print(f"[{status}] {case['query']}")
        if not ok:
            failures.append({"query": case["query"], "response_excerpt": response[:120],
                             "tools_used": tools_used})

    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else 0
    return {
        "total": len(cases),
        "passed": passed,
        "task_completion_rate": passed / len(cases),
        "latency_avg_ms": round(sum(latencies) / len(latencies), 1),
        "latency_p95_ms": round(p95, 1),
        "failures": failures,
    }


def write_task_report(result: Dict[str, Any]) -> str:
    lines = [
        "# 任务完成率评测报告", "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        f"- 任务完成率：**{result['task_completion_rate']:.1%}**（{result['passed']}/{result['total']}）",
        f"- 端到端延迟：均值 {result['latency_avg_ms']}ms / P95 {result['latency_p95_ms']}ms",
        "",
    ]
    for f_ in result["failures"]:
        lines.append(f"- 未完成：{f_['query']}")
        lines.append(f"  - 实际工具：{f_['tools_used'] or '无'}；回答摘录：{f_['response_excerpt']}")
    report = "\n".join(lines) + "\n"
    with open(TASK_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main():
    result = eval_task()
    print()
    print(write_task_report(result))
    print(f"report written to {TASK_REPORT_PATH}")


if __name__ == "__main__":
    main()
