"""
多维度指标评测：检索质量 / 工具调用 / 生成忠实度

用法：
    python -m evaluation.run_metrics_eval retrieval     # 检索质量（需已建索引，无需 API Key）
    python -m evaluation.run_metrics_eval tool          # 工具调用准确率（需 API Key）
    python -m evaluation.run_metrics_eval faithfulness  # 生成忠实度（需 API Key + 已建索引）
    python -m evaluation.run_metrics_eval all           # 全部

产物：evaluation/metrics_report.md
指标定义：
- 检索质量：Recall@1 / Recall@3（期望文档是否出现在 Top-K）、MRR（期望文档排名倒数均值）、检索延迟
- 工具调用：选择准确率（必需工具全部被调用且禁止工具未被调用）、参数抽取准确率（关键参数与期望一致）
- 生成忠实度：LLM-as-judge 把回答拆分为事实性声明，逐条对照检索到的参考材料核验；
  忠实度 = 被支持的声明占比，幻觉率 = 1 - 忠实度（与 RAGAS Faithfulness 同思路的轻量实现）
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from evaluation.golden_utils import (
    aggregate_faithfulness,
    judge_tool_case,
    rank_of_expected,
)
from log_setup import get_logger

logger = get_logger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(EVAL_DIR, "metrics_report.md")


def _load_jsonl(name: str) -> List[Dict[str, Any]]:
    path = os.path.join(EVAL_DIR, name)
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------- 检索质量

def eval_retrieval() -> Optional[Dict[str, Any]]:
    """对 retrieval_golden 逐条检索，统计 Recall@1/3、MRR 与检索延迟"""
    from retrieval.vector_store import _get_context, search_kb_with_sources
    from config import KB_INDEX_PATH

    if not os.path.exists(os.path.join(KB_INDEX_PATH, "index.faiss")):
        print("[retrieval] 知识库索引不存在，跳过（先运行 python -m retrieval.build）")
        return None

    cases = _load_jsonl("retrieval_golden.jsonl")
    # 预热：本地嵌入模型首次加载耗时数秒，属一次性冷启动，不计入延迟统计
    search_kb_with_sources("预热查询", k=1)

    ranks, latencies, failures = [], [], []
    for case in cases:
        t0 = time.perf_counter()
        results = search_kb_with_sources(case["query"], k=3)
        latency_ms = (time.perf_counter() - t0) * 1000
        sources = [s for s, _, _ in results]
        rank = rank_of_expected(sources, case["expected_doc"])
        ranks.append(rank)
        latencies.append(latency_ms)
        status = "OK " if rank else "MISS"
        print(f"[{status}] rank={rank if rank else '-'}  {case['query']}")
        if rank is None:
            failures.append({"query": case["query"], "expected": case["expected_doc"],
                             "got": sources})

    hit3 = sum(1 for r in ranks if r and r <= 3)
    mrr = sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else 0
    return {
        "total": len(cases),
        "recall_at_1": sum(1 for r in ranks if r == 1) / len(cases),
        "recall_at_3": hit3 / len(cases),
        "mrr": mrr,
        "latency_avg_ms": round(sum(latencies) / len(latencies), 1),
        "latency_p95_ms": round(p95, 1),
        "failures": failures,
    }


# ---------------------------------------------------------------- 工具调用

def _make_recorder(tool, store: Dict[str, List[dict]]):
    """包装业务工具：保持完整工具 schema（bind_tools 需要），调用透传并记录参数"""
    from langchain_core.tools import StructuredTool

    def _run(**kwargs):
        store.setdefault(tool.name, []).append(kwargs or {})
        return tool.invoke(kwargs)

    return StructuredTool.from_function(
        func=_run, name=tool.name, description=tool.description,
        args_schema=tool.args_schema,
    )


def eval_tool() -> Optional[Dict[str, Any]]:
    """对 tool_golden 逐例直跑对应专家，统计工具选择与参数抽取准确率"""
    from multi_agent_customer_service import initialize_agents
    from llm import get_llm
    import multi_agents.billing_agent as billing_mod
    import multi_agents.tech_agent as tech_mod

    if get_llm() is None:
        print("[tool] 未配置 API Key，跳过（在 .env 配置后重跑）")
        return None

    cases = _load_jsonl("tool_golden.jsonl")
    tool_ok_count, arg_cases, arg_ok_count = 0, 0, 0
    failures = []

    for i, case in enumerate(cases):
        agent_mod = billing_mod if case["agent"] == "billing_agent" else tech_mod
        recorded: Dict[str, List[dict]] = {}
        # 用记录器替换专家模块内的工具引用（进程内、仅评测期间生效）
        patched = []
        for attr in ("query_order", "create_ticket"):
            if hasattr(agent_mod, attr):
                original = getattr(agent_mod, attr)
                setattr(agent_mod, attr, _make_recorder(original, recorded))
                patched.append((attr, original))
        try:
            agents = initialize_agents()
            agent = agents[case["agent"]]
            state = {
                "customer_query": case["query"],
                "tools_used": [],
                "persisted_dialogue": [],
                "handoff_count": 0,
                "session_id": f"eval-tool-{i}",
            }
            t0 = time.perf_counter()
            agent.process(state)
            latency_ms = (time.perf_counter() - t0) * 1000
        finally:
            for attr, original in patched:
                setattr(agent_mod, attr, original)

        used = state.get("tools_used", [])
        tool_ok, arg_ok = judge_tool_case(used, recorded, case)
        tool_ok_count += tool_ok
        if arg_ok is not None:
            arg_cases += 1
            arg_ok_count += arg_ok
        status = "OK " if tool_ok and arg_ok is not False else "MISS"
        print(f"[{status}] tools={used or '无'}  {case['query']}")
        if not (tool_ok and arg_ok is not False):
            failures.append({"query": case["query"], "expected_tools": case["required_tools"],
                             "forbidden_tools": case["forbidden_tools"], "used": used,
                             "recorded_args": recorded, "expected_args": case["expected_args"]})

    return {
        "total": len(cases),
        "tool_selection_accuracy": tool_ok_count / len(cases),
        "arg_extraction_accuracy": (arg_ok_count / arg_cases) if arg_cases else None,
        "failures": failures,
    }


# ---------------------------------------------------------------- 生成忠实度

class _Claim(BaseModel):
    claim: str = Field(description="回答中的一条事实性声明")
    supported: bool = Field(description="该声明是否被参考材料支持")


class _ClaimCheck(BaseModel):
    claims: List[_Claim] = Field(default_factory=list, description="事实性声明列表")


_JUDGE_SYSTEM = (
    "你是事实核验助手。把回答拆分为独立的事实性声明（数字、参数、政策条款等），"
    "逐条判断是否被参考材料支持。参考材料未提及的信息视为不支持；"
    "客服礼仪、建议性话语和引导性话术不算事实性声明；"
    "对话过程引用（如「您之前问过」「这是第四次确认」「上次系统报错」）属于会话元信息，"
    "同样不算事实性声明，忽略。"
)


def parse_claims_text(text: str) -> List[Dict[str, Any]]:
    """从纯文本回复中解析声明 JSON 数组（结构化输出降级路径）"""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    data = json.loads(m.group(0))
    return [{"claim": str(d.get("claim", "")), "supported": bool(d.get("supported"))}
            for d in data if isinstance(d, dict) and d.get("claim")]


def _judge_claims(llm, references: str, answer: str) -> List[Dict[str, Any]]:
    """裁判一条回答：返回声明列表。结构化输出失败时回退纯文本 JSON 解析"""
    user = f"参考材料：\n{references}\n\n回答：\n{answer}"
    try:
        verdict = llm.with_structured_output(_ClaimCheck).invoke([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ])
        return [c.model_dump() for c in verdict.claims]
    except Exception as e:
        logger.warning(f"[faithfulness] structured judge failed, fallback to text: {e}")

    reply = llm.invoke([
        {"role": "system", "content": _JUDGE_SYSTEM +
            '只输出 JSON 数组，每项形如 {"claim": "声明内容", "supported": true}，不要输出其他内容。'},
        {"role": "user", "content": user},
    ])
    content = reply.content if isinstance(reply.content, str) else str(reply.content)
    return parse_claims_text(content)


def eval_faithfulness() -> Optional[Dict[str, Any]]:
    """对检索 golden 的问题逐条跑完整图，用 LLM-as-judge 核验回答与参考材料的一致性"""
    from app import get_graph
    from llm import get_llm
    from retrieval.vector_store import search_kb_with_sources

    llm = get_llm()
    if llm is None:
        print("[faithfulness] 未配置 API Key，跳过（在 .env 配置后重跑）")
        return None

    cases = _load_jsonl("retrieval_golden.jsonl")
    graph = get_graph()
    run_id = str(int(time.time()))  # 会话按次隔离，避免 checkpointer 历史跨轮累积
    per_question, latencies = [], []
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        final = graph.invoke(
            {"customer_query": case["query"], "session_id": f"eval-faith-{run_id}-{i}"},
            {"configurable": {"thread_id": f"eval-faith-{run_id}-{i}"}},
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        answer = final.get("response", "")
        # 参考材料截断：控制 judge 输入规模，避免长上下文导致超时
        references = "\n\n".join(
            chunk[:600] for _, _, chunk in search_kb_with_sources(case["query"], k=3)
        )
        if not references.strip():
            per_question.append({"query": case["query"], "score": None, "reason": "no reference"})
            print(f"[SKIP] 无参考材料  {case['query']}")
            continue

        claims = None
        for attempt in range(2):  # 超时等瞬态失败重试一次
            try:
                claims = _judge_claims(llm, references, answer)
                break
            except Exception as e:
                reason = f"judge error: {e}"
                logger.warning(f"[faithfulness] attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        if claims is None:
            per_question.append({"query": case["query"], "score": None, "reason": reason})
            print(f"[SKIP] judge 失败  {case['query']}")
            continue

        score = aggregate_faithfulness(claims)
        per_question.append({"query": case["query"], "score": score, "claims": claims})
        if score is None:
            print(f"[SKIP] 无事实性声明  {case['query']}")
        else:
            print(f"[{'OK ' if score == 1 else 'HALLUCINATION'}] faithfulness={score:.2f}  {case['query']}")

    scored = [q["score"] for q in per_question if q.get("score") is not None]
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else 0
    faithfulness = sum(scored) / len(scored) if scored else None
    return {
        "total": len(cases),
        "faithfulness": round(faithfulness, 4) if faithfulness is not None else None,
        "hallucination_rate": round(1 - faithfulness, 4) if faithfulness is not None else None,
        "e2e_latency_avg_ms": round(sum(latencies) / len(latencies), 1),
        "e2e_latency_p95_ms": round(p95, 1),
        "detail": per_question,
    }


# ---------------------------------------------------------------- 报告

def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "跳过"


def write_report(retrieval: Optional[dict], tool: Optional[dict],
                 faithfulness: Optional[dict]) -> str:
    lines = ["# 多维度指标评测报告", "",
             f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    if retrieval:
        lines += [
            "## 检索质量", "",
            f"- Recall@1：**{_fmt_pct(retrieval['recall_at_1'])}**",
            f"- Recall@3：**{_fmt_pct(retrieval['recall_at_3'])}**",
            f"- MRR：**{retrieval['mrr']:.3f}**",
            f"- 检索延迟（已剔除首次加载模型的冷启动）：均值 {retrieval['latency_avg_ms']}ms / P95 {retrieval['latency_p95_ms']}ms",
            "",
        ]
        for f_ in retrieval["failures"]:
            lines.append(f"- 未命中：{f_['query']}（期望 {f_['expected']}，实际 {f_['got']}）")
        lines.append("")
    else:
        lines += ["## 检索质量", "", "- 跳过（索引未构建或依赖缺失，先运行 python -m retrieval.build）", ""]

    if tool:
        lines += [
            "## 工具调用", "",
            f"- 工具选择准确率：**{_fmt_pct(tool['tool_selection_accuracy'])}**",
            f"- 参数抽取准确率：**{_fmt_pct(tool['arg_extraction_accuracy'])}**",
            "",
        ]
        for f_ in tool["failures"]:
            lines.append(f"- 异常：{f_['query']}（期望 {f_['expected_tools']}，实际 {f_['used']}，"
                         f"捕获参数 {f_['recorded_args'] or '无'}）")
        lines.append("")
    else:
        lines += ["## 工具调用", "", "- 跳过（未配置 API Key）", ""]

    if faithfulness and faithfulness.get("faithfulness") is not None:
        lines += [
            "## 生成忠实度（LLM-as-judge）", "",
            f"- 忠实度：**{_fmt_pct(faithfulness['faithfulness'])}**",
            f"- 幻觉率（未支持声明占比）：**{_fmt_pct(faithfulness['hallucination_rate'])}**",
            f"- 端到端延迟：均值 {faithfulness['e2e_latency_avg_ms']}ms / P95 {faithfulness['e2e_latency_p95_ms']}ms",
            "",
        ]
        for q in faithfulness["detail"]:
            if q.get("score") is not None and q["score"] < 1:
                bad = [c["claim"] for c in q.get("claims", []) if not c["supported"]]
                lines.append(f"- 疑似幻觉：{q['query']} -> {'；'.join(bad) or '（无未支持声明明细）'}")
            elif q.get("score") is None:
                lines.append(f"- 未核验：{q['query']}（{q.get('reason', '无事实性声明')}）")
        lines.append("")
    else:
        lines += ["## 生成忠实度（LLM-as-judge）", "",
                  "- 跳过或未产出有效核验（需 API Key + 已建索引；若两者都在，检查上方 judge 日志）", ""]
    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="多维度指标评测")
    parser.add_argument("target", nargs="?", default="all",
                        choices=["retrieval", "tool", "faithfulness", "all"])
    args = parser.parse_args()

    retrieval = tool = faith = None
    if args.target in ("retrieval", "all"):
        retrieval = eval_retrieval()
    if args.target in ("tool", "all"):
        tool = eval_tool()
    if args.target in ("faithfulness", "all"):
        faith = eval_faithfulness()

    report = write_report(retrieval, tool, faith)
    print(f"\n{report}")
    print(f"report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
