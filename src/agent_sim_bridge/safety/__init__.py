"""Safety subsystem — constraints, real-time monitoring, and boundary checking.

Provides a layered safety architecture:

1. **Constraints** (:mod:`~agent_sim_bridge.safety.constraints`) — declarative
   per-action validation rules.
2. **Monitor** (:mod:`~agent_sim_bridge.safety.monitor`) — stateful per-episode
   violation accumulation with emergency stop.
3. **Boundaries** (:mod:`~agent_sim_bridge.safety.boundaries`) — geometric
   workspace limits validated against observation/position vectors.
"""
from __future__ import annotations

from agent_sim_bridge.safety.boundaries import (
    BoundaryChecker,
    BoundaryDefinition,
    BoundaryViolation,
)
from agent_sim_bridge.safety.constraints import (
    ConstraintType,
    SafetyChecker,
    SafetyConstraint,
    SafetyViolation,
    ViolationSeverity,
)
from agent_sim_bridge.safety.monitor import MonitoredStep, SafetyMonitor

__all__ = [
    "SafetyConstraint",
    "SafetyChecker",
    "SafetyViolation",
    "ConstraintType",
    "ViolationSeverity",
    "SafetyMonitor",
    "MonitoredStep",
    "BoundaryDefinition",
    "BoundaryChecker",
    "BoundaryViolation",
]
