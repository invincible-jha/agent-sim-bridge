"""Shared bootstrap for agent-sim-bridge benchmarks."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_SRC = _REPO_ROOT / "src"
_BENCHMARKS = _REPO_ROOT / "benchmarks"

for _path in [str(_SRC), str(_BENCHMARKS)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent_sim_bridge.gap.estimator import GapDimension, GapEstimator, GapMetric
from agent_sim_bridge.simulation.scenario import Scenario, ScenarioManager

__all__ = ["GapDimension", "GapEstimator", "GapMetric", "Scenario", "ScenarioManager"]
