"""
异常恢复率评测：运行全部故障注入场景，统计按设计降级/自愈的比例。
全部离线可跑：python -m evaluation.run_resilience_eval
"""

from evaluation.resilience_scenarios import SCENARIOS


def main():
    print("异常注入恢复演练\n" + "-" * 56)
    passed = 0
    details = []
    for scenario in SCENARIOS:
        try:
            name, ok, detail = scenario()
        except Exception as e:  # 场景自身抛异常即视为未恢复
            name, ok, detail = scenario.__name__.replace("scenario_", ""), False, f"scenario crashed: {e}"
        passed += ok
        details.append((name, ok, detail))
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}")

    rate = passed / len(SCENARIOS)
    print("-" * 56)
    print(f"异常恢复率: {passed}/{len(SCENARIOS)} = {rate:.1%}")
    if rate < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
