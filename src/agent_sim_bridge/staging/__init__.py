"""Agent staging environment — simulate users, tools, and chaos for pre-production testing.

This subpackage provides a full staging environment that lets you run
an agent function against simulated users and tools with configurable
chaos injection (network partitions, latency spikes, model degradation).

Submodules
----------
- ``environment``        StagingEnvironment, StagingReport
- ``simulated_users``    SimulatedUser, UserProfile
- ``simulated_tools``    SimulatedTool, ToolBehavior
- ``chaos``              ChaosEngine, ChaosConfig
- ``results``            StagingTestResult, aggregation helpers
"""
from __future__ import annotations

from agent_sim_bridge.staging.chaos import ChaosConfig, ChaosEngine
from agent_sim_bridge.staging.environment import StagingEnvironment, StagingReport
from agent_sim_bridge.staging.results import StagingTestResult
from agent_sim_bridge.staging.simulated_tools import SimulatedTool, ToolBehavior
from agent_sim_bridge.staging.simulated_users import SimulatedUser, UserProfile

__all__ = [
    "ChaosConfig",
    "ChaosEngine",
    "StagingEnvironment",
    "StagingReport",
    "StagingTestResult",
    "SimulatedTool",
    "ToolBehavior",
    "SimulatedUser",
    "UserProfile",
]
