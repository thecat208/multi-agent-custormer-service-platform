"""
评测脚本：意图路由正确率 + 越界拒答正确率
用法：python -m evaluation.run_eval [--limit N] [--out evaluation/report.md]
需要 .env 配置 OPENAI_API_KEY（评测的是真实 LLM 分类链路）。
产出真实指标写入 report.md；全部通过时退出码 0，否则 1（可接 CI）。
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime

from config import OPENAI_API_KEY, OPENAI_MODEL
from multi_agent_customer_service import make_graph

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.jsonl")


def load_cases(path: str = GOLDEN_PATH, limit: int = 0):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def run_eval(cases):
    graph = make_graph()  # 评测无需 checkpointer，用例之间相互独立
    results = []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        expected = case["query_type"]
        t0 = time.time()
        try:
            state = graph.invoke({"customer_query": query})
            got = state.get("query_type", "")
            agent = state.get("current_agent", "")
            error = ""
        except Exception as e:
            got, agent, error = "", "", str(e)
        latency_ms = (time.time() - t0) * 1000
        results.append({
            "index": i,
            "query": query,
            "expected": expected,
            "got": got,
            "agent": agent,
            "passed": got == expected,
            "latency_ms": round(latency_ms, 1),
            "error": error,
        })
        status = "PASS" if got == expected else "FAIL"
        print(f"[{i:>2}/{len(cases)}] {status}  expected={expected:<18} got={got:<18} {latency_ms:>6.0f}ms  {query}")
    return results


def summarize(results):
    total = len(results)
    passed = sum(r["passed"] for r in results)
    refusals = [r for r in results if r["expected"] == "out_of_scope"]
    refusal_recall = (
        sum(r["passed"] for r in refusals) / len(refusals) if refusals else 0.0
    )
    by_class = Counter(r["expected"] for r in results)
    per_class = {}
    for cls, n in by_class.items():
        sub = [r for r in results if r["expected"] == cls]
        per_class[cls] = {
            "total": n,
            "passed": sum(r["passed"] for r in sub),
            "accuracy": round(sum(r["passed"] for r in sub) / n, 3),
        }
    latencies = sorted(r["latency_ms"] for r in results)
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0
    return {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total, 3) if total else 0.0,
        "refusal_recall": round(refusal_recall, 3),
        "latency_avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(p95, 1),
        "per_class": per_class,
    }


def write_report(metrics, results, out_path: str):
    lines = [
        "# 意图路由评测报告",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：{OPENAI_MODEL}（硅基流动 OpenAI 兼容端点）",
        f"- 用例：{metrics['total']} 条（自建 golden set）",
        "",
        "## 总体指标",
        "",
        f"- 路由正确率：**{metrics['passed']}/{metrics['total']} = {metrics['accuracy']:.1%}**",
        f"- 越界拒答正确率：**{metrics['refusal_recall']:.1%}**",
        f"- 端到端延迟：平均 {metrics['latency_avg_ms']}ms / P95 {metrics['latency_p95_ms']}ms",
        "",
        "## 分类别正确率",
        "",
        "| 类别 | 通过/总数 | 正确率 |",
        "|---|---|---|",
    ]
    for cls, m in metrics["per_class"].items():
        lines.append(f"| {cls} | {m['passed']}/{m['total']} | {m['accuracy']:.1%} |")

    failures = [r for r in results if not r["passed"]]
    if failures:
        lines += ["", "## 失败用例", ""]
        for r in failures:
            lines.append(
                f"- [{r['index']}] {r['query']} → 期望 {r['expected']}，实际 {r['got'] or r['error']}"
            )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nreport written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.md"))
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise SystemExit("未配置 OPENAI_API_KEY：评测的是真实 LLM 链路，请先在 .env 配置 Key")

    cases = load_cases(limit=args.limit)
    print(f"running {len(cases)} golden cases ...\n")
    results = run_eval(cases)
    metrics = summarize(results)

    print(f"\n路由正确率: {metrics['passed']}/{metrics['total']} = {metrics['accuracy']:.1%}")
    print(f"越界拒答正确率: {metrics['refusal_recall']:.1%}")
    print(f"延迟: 平均 {metrics['latency_avg_ms']}ms / P95 {metrics['latency_p95_ms']}ms")

    write_report(metrics, results, args.out)
    raise SystemExit(0 if metrics["accuracy"] == 1.0 else 1)


if __name__ == "__main__":
    main()
