"""
单次任务平均成本评测：端到端跑 golden 问题，经回调统计 token 消耗，按单价折算。

用法：python -m evaluation.run_cost_eval（需 API Key；建议已建索引以覆盖检索问答场景）
单价配置（.env，单位：元 / 百万 token）：
    PRICE_INPUT_PER_M  默认 2.0
    PRICE_OUTPUT_PER_M 默认 8.0
产物：evaluation/cost_report.md
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from evaluation.golden_utils import compute_cost
from evaluation.run_metrics_eval import _load_jsonl
from log_setup import get_logger
from config import PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M

logger = get_logger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
COST_REPORT_PATH = os.path.join(EVAL_DIR, "cost_report.md")


class TokenUsageHandler(BaseCallbackHandler):
    """累计整次图执行中所有 LLM 调用的 token 用量"""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response, **kwargs) -> None:
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        self.input_tokens += usage.get("prompt_tokens", 0) or 0
        self.output_tokens += usage.get("completion_tokens", 0) or 0
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def eval_cost() -> Optional[Dict[str, Any]]:
    from app import get_graph
    from llm import get_llm

    if get_llm() is None:
        print("[cost] 未配置 API Key，跳过（在 .env 配置后重跑）")
        return None

    cases = _load_jsonl("task_golden.jsonl")
    graph = get_graph()
    run_id = str(int(time.time()))  # 会话按次隔离，避免 checkpointer 历史跨轮累积
    per_task: List[Dict[str, Any]] = []

    for i, case in enumerate(cases):
        handler = TokenUsageHandler()
        graph.invoke(
            {"customer_query": case["query"], "session_id": f"eval-cost-{run_id}-{i}"},
            {"configurable": {"thread_id": f"eval-cost-{run_id}-{i}"}, "callbacks": [handler]},
        )
        cost = compute_cost(handler.input_tokens, handler.output_tokens,
                            PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M)
        per_task.append({
            "query": case["query"],
            "input_tokens": handler.input_tokens,
            "output_tokens": handler.output_tokens,
            "total_tokens": handler.total_tokens,
            "llm_calls": handler.calls,
            "cost_yuan": round(cost, 6),
        })
        print(f"[{i + 1}/{len(cases)}] tokens={handler.total_tokens:>6} "
              f"(in {handler.input_tokens} / out {handler.output_tokens}) "
              f"calls={handler.calls}  cost=¥{cost:.4f}  {case['query']}")

    n = len(per_task)
    avg_in = sum(t["input_tokens"] for t in per_task) / n
    avg_out = sum(t["output_tokens"] for t in per_task) / n
    avg_cost = sum(t["cost_yuan"] for t in per_task) / n
    return {
        "total": n,
        "avg_input_tokens": round(avg_in, 1),
        "avg_output_tokens": round(avg_out, 1),
        "avg_llm_calls": round(sum(t["llm_calls"] for t in per_task) / n, 1),
        "avg_cost_yuan": round(avg_cost, 6),
        "price_input_per_m": PRICE_INPUT_PER_M,
        "price_output_per_m": PRICE_OUTPUT_PER_M,
        "detail": per_task,
    }


def write_cost_report(result: Dict[str, Any]) -> str:
    lines = [
        "# 单次任务成本评测报告", "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 单价：输入 ¥{result['price_input_per_m']}/M tokens，输出 ¥{result['price_output_per_m']}/M tokens"
        "（可在 .env 的 PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M 调整）", "",
        f"- 单次任务平均 token：输入 **{result['avg_input_tokens']}** + 输出 **{result['avg_output_tokens']}**"
        f"（LLM 调用 {result['avg_llm_calls']} 次/任务）",
        f"- 单次任务平均成本：**¥{result['avg_cost_yuan']}**",
        "",
    ]
    for t in result["detail"]:
        lines.append(f"- {t['query']}：{t['total_tokens']} tokens，¥{t['cost_yuan']}")
    report = "\n".join(lines) + "\n"
    with open(COST_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main():
    result = eval_cost()
    if result is None:
        return
    print()
    print(write_cost_report(result))
    print(f"report written to {COST_REPORT_PATH}")


if __name__ == "__main__":
    main()
