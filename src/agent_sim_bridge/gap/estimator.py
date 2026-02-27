"""Sim-to-real gap estimation engine.

Provides dataclasses and a class for computing statistical distance measures
between paired simulation and real-world distributions across multiple
dimensions.

Classes
-------
GapDimension
    Container for one named sim/real distribution pair.
GapMetric
    Enumeration of supported statistical distance metrics.
DimensionGap
    Result of evaluating one metric on one dimension.
GapEstimator
    Orchestrates multi-metric, multi-dimension gap estimation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from agent_sim_bridge.gap.statistics import (
    jensen_shannon_divergence,
    kl_divergence,
    maximum_mean_discrepancy,
    normalize_distribution,
    wasserstein_distance_1d,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds for gap interpretation
# ---------------------------------------------------------------------------

# Thresholds are (low_upper, medium_upper); anything above medium_upper → "high".
_THRESHOLDS: dict[str, tuple[float, float]] = {
    "KL_DIVERGENCE":    (0.1,  0.5),
    "WASSERSTEIN":      (0.05, 0.2),
    "MMD":              (0.05, 0.2),
    "JENSEN_SHANNON":   (0.05, 0.2),
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class GapMetric(Enum):
    """Statistical distance metric to use when comparing distributions.

    Members
    -------
    KL_DIVERGENCE
        Kullback-Leibler divergence KL(sim || real).  Non-symmetric.
    WASSERSTEIN
        Wasserstein-1 / Earth Mover's Distance.  Symmetric.
    MMD
        Maximum Mean Discrepancy with RBF kernel.  Symmetric.
    JENSEN_SHANNON
        Jensen-Shannon divergence.  Symmetric and bounded in [0, ln(2)].
    """

    KL_DIVERGENCE = "KL_DIVERGENCE"
    WASSERSTEIN = "WASSERSTEIN"
    MMD = "MMD"
    JENSEN_SHANNON = "JENSEN_SHANNON"


@dataclass(frozen=True)
class GapDimension:
    """A named pair of sim and real distributions for one observable dimension.

    Attributes
    ----------
    name:
        Human-readable label for this dimension (e.g. ``"joint_velocity_0"``).
    sim_distribution:
        Empirical distribution collected in simulation.
    real_distribution:
        Empirical distribution collected on the real system.
    """

    name: str
    sim_distribution: list[float]
    real_distribution: list[float]


@dataclass(frozen=True)
class DimensionGap:
    """The gap value for one metric applied to one dimension.

    Attributes
    ----------
    dimension_name:
        The name of the :class:`GapDimension` this result belongs to.
    metric:
        The :class:`GapMetric` used to compute the value.
    value:
        The computed distance value.
    interpretation:
        Qualitative severity: ``"low"``, ``"medium"``, or ``"high"``.
    """

    dimension_name: str
    metric: GapMetric
    value: float
    interpretation: str


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class GapEstimator:
    """Compute sim-to-real distribution gaps across multiple metrics and dimensions.

    Parameters
    ----------
    metrics:
        List of :class:`GapMetric` values to compute.  Defaults to all four.

    Example
    -------
    ::

        estimator = GapEstimator()
        dim = GapDimension(
            name="torque",
            sim_distribution=[0.1, 0.3, 0.4, 0.2],
            real_distribution=[0.15, 0.25, 0.35, 0.25],
        )
        gaps = estimator.estimate_dimension(dim)
        score = estimator.overall_gap_score({"torque": gaps})
    """

    def __init__(self, metrics: list[GapMetric] | None = None) -> None:
        self._metrics: list[GapMetric] = metrics if metrics is not None else list(GapMetric)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def estimate_dimension(self, dim: GapDimension) -> list[DimensionGap]:
        """Compute all configured metrics for a single dimension.

        Parameters
        ----------
        dim:
            The :class:`GapDimension` to analyse.

        Returns
        -------
        list[DimensionGap]
            One :class:`DimensionGap` per configured metric.
        """
        results: list[DimensionGap] = []
        for metric in self._metrics:
            value = self._compute_metric(metric, dim)
            interpretation = self._interpret(metric, value)
            results.append(
                DimensionGap(
                    dimension_name=dim.name,
                    metric=metric,
                    value=value,
                    interpretation=interpretation,
                )
            )
            logger.debug(
                "Gap[%s, %s] = %.6f (%s)",
                dim.name,
                metric.value,
                value,
                interpretation,
            )
        return results

    def estimate_all(
        self, dimensions: list[GapDimension]
    ) -> dict[str, list[DimensionGap]]:
        """Compute all configured metrics for multiple dimensions.

        Parameters
        ----------
        dimensions:
            Sequence of :class:`GapDimension` objects.

        Returns
        -------
        dict[str, list[DimensionGap]]
            Mapping from dimension name to its list of gap results.
        """
        return {dim.name: self.estimate_dimension(dim) for dim in dimensions}

    def overall_gap_score(
        self, dimension_gaps: dict[str, list[DimensionGap]]
    ) -> float:
        """Compute a single weighted average gap score in [0, 1].

        Each :class:`DimensionGap` value is first normalised to [0, 1] using
        the ``"high"`` threshold for the respective metric before being
        averaged.  This makes the final score interpretable regardless of
        the differing natural scales of the individual metrics.

        Parameters
        ----------
        dimension_gaps:
            Output of :meth:`estimate_all`.

        Returns
        -------
        float
            Weighted average gap score ∈ [0, 1].  Returns 0.0 if there are
            no results.
        """
        normalised_values: list[float] = []
        for gaps in dimension_gaps.values():
            for gap in gaps:
                high_threshold = _THRESHOLDS[gap.metric.value][1]
                normalised = min(1.0, gap.value / high_threshold) if high_threshold > 0.0 else 0.0
                normalised_values.append(normalised)

        if not normalised_values:
            return 0.0
        return sum(normalised_values) / len(normalised_values)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_metric(self, metric: GapMetric, dim: GapDimension) -> float:
        """Dispatch to the correct statistical function for *metric*.

        Parameters
        ----------
        metric:
            The metric to compute.
        dim:
            Source of sim/real distributions.

        Returns
        -------
        float
            The computed distance value.
        """
        sim = dim.sim_distribution
        real = dim.real_distribution

        if metric is GapMetric.KL_DIVERGENCE:
            return kl_divergence(sim, real)
        if metric is GapMetric.WASSERSTEIN:
            return wasserstein_distance_1d(sim, real)
        if metric is GapMetric.MMD:
            return maximum_mean_discrepancy(sim, real)
        if metric is GapMetric.JENSEN_SHANNON:
            return jensen_shannon_divergence(sim, real)
        # Unreachable for a well-typed GapMetric, but guards future additions.
        raise ValueError(f"Unknown GapMetric: {metric!r}")  # pragma: no cover

    def _interpret(self, metric: GapMetric, value: float) -> str:
        """Map a raw distance value to a qualitative severity label.

        Parameters
        ----------
        metric:
            The metric whose thresholds to apply.
        value:
            The computed distance.

        Returns
        -------
        str
            ``"low"``, ``"medium"``, or ``"high"``.
        """
        low_upper, medium_upper = _THRESHOLDS[metric.value]
        if value <= low_upper:
            return "low"
        if value <= medium_upper:
            return "medium"
        return "high"

    def __repr__(self) -> str:
        metric_names = [m.value for m in self._metrics]
        return f"GapEstimator(metrics={metric_names})"
