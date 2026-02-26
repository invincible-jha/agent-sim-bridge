"""Environment abstractions for sim-to-real bridge.

Provides the abstract base, concrete sim/real wrappers, and the adapter
that normalises the two interfaces so agent code is portable across
simulation and physical deployments.
"""
from __future__ import annotations

from agent_sim_bridge.environment.adapter import EnvironmentAdapter
from agent_sim_bridge.environment.base import Environment, EnvironmentInfo, StepResult
from agent_sim_bridge.environment.real_env import RealityEnvironment
from agent_sim_bridge.environment.sim_env import SimulationEnvironment

__all__ = [
    "Environment",
    "EnvironmentInfo",
    "StepResult",
    "SimulationEnvironment",
    "RealityEnvironment",
    "EnvironmentAdapter",
]
