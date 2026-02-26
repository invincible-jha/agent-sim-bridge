"""Simulation execution, trajectory recording, replay, and scenario management."""
from __future__ import annotations

from agent_sim_bridge.simulation.recorder import TrajectoryRecorder, TrajectoryStep
from agent_sim_bridge.simulation.replay import TrajectoryReplay
from agent_sim_bridge.simulation.sandbox import ExecutionResult, SimulationSandbox
from agent_sim_bridge.simulation.scenario import Scenario, ScenarioManager

__all__ = [
    "SimulationSandbox",
    "ExecutionResult",
    "TrajectoryRecorder",
    "TrajectoryStep",
    "TrajectoryReplay",
    "Scenario",
    "ScenarioManager",
]
