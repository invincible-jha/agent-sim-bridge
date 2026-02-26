"""Metrics subsystem — sim-to-real gap analysis and performance tracking.

Provides:

* **gap** — :class:`SimRealGap` and :class:`GapReport` for quantifying the
  sim-to-real discrepancy.
* **performance** — :class:`PerformanceTracker` for lightweight in-memory
  metric accumulation and statistical summaries.
"""
from __future__ import annotations

from agent_sim_bridge.metrics.gap import GapReport, SimRealGap
from agent_sim_bridge.metrics.performance import MetricRecord, PerformanceTracker

__all__ = [
    "SimRealGap",
    "GapReport",
    "PerformanceTracker",
    "MetricRecord",
]
