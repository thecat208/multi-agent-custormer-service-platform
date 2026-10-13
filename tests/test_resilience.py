"""异常注入场景全部按设计恢复（离线，并入 CI 回归）"""

import pytest

from evaluation.resilience_scenarios import SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_scenario_recovers(scenario):
    name, ok, detail = scenario()
    assert ok, f"{name} 未按设计恢复: {detail}"
