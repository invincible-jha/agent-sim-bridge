"""Sim-to-real gap estimation engine.

This package provides distribution-based tools for quantifying the discrepancy
between simulation and real-world data across multiple observable dimensions.

Unlike the trajectory-level ``metrics.gap`` module (which measures MAE/RMSE
between paired rollouts), this package works on *distributions* — useful when
you have population-level samples from simulation and the real world but not
necessarily step-aligned pairs.

Submodules
----------
statistics:
    Pure Python implementations of KL divergence, Wasserstein-1 distance, MMD,
    and Jensen-Shannon divergence.  No numpy or scipy dependency.
estimator:
    :class:`GapEstimator` orchestrates multi-metric, multi-dimension analysis.
    :class:`GapDimension`, :class:`GapMetric`, and :class:`DimensionGap` are
    the core data types.
report:
    :class:`GapReporter` generates :class:`GapReport` objects and serialises
    them as plain text, JSON, or Markdown.
"""
from __future__ import annotations

from agent_sim_bridge.gap.estimator import (
    DimensionGap,
    GapDimension,
    GapEstimator,
    GapMetric,
)
from agent_sim_bridge.gap.report import GapReport, GapReporter
from agent_sim_bridge.gap.statistics import (
    descriptive_stats,
    jensen_shannon_divergence,
    kl_divergence,
    maximum_mean_discrepancy,
    normalize_distribution,
    wasserstein_distance_1d,
)

__all__: list[str] = [
    # estimator
    "GapDimension",
    "GapMetric",
    "DimensionGap",
    "GapEstimator",
    # report
    "GapReport",
    "GapReporter",
    # statistics
    "kl_divergence",
    "wasserstein_distance_1d",
    "maximum_mean_discrepancy",
    "jensen_shannon_divergence",
    "normalize_distribution",
    "descriptive_stats",
]
