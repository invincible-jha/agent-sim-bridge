"""Production readiness scorer.

Computes a composite readiness score (0-100) from four dimensions:

- **Reliability** (weight 30 %): derived from uptime fraction and error rate.
- **Safety** (weight 30 %): derived from violation count over sample size.
- **Performance** (weight 25 %): derived from latency p50 and p99 values.
- **Cost** (weight 15 %): derived from cost-per-task relative to a budget.

Each dimension is scored 0-100 independently, then combined into a weighted
average.  Interpretation bands: 0-49 = Not Ready, 50-69 = Needs Work,
70-84 = Mostly Ready, 85-100 = Production Ready.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ReadinessGrade(str, Enum):
    """Qualitative grade for a readiness score."""

    NOT_READY = "not_ready"
    NEEDS_WORK = "needs_work"
    MOSTLY_READY = "mostly_ready"
    PRODUCTION_READY = "production_ready"


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadinessInput:
    """Raw metrics used to compute the readiness score.

    Attributes
    ----------
    uptime_fraction:
        Fraction of time the agent was operational, in [0.0, 1.0].
    error_rate:
        Fraction of requests that resulted in errors, in [0.0, 1.0].
    violation_count:
        Number of safety violations observed.
    sample_size:
        Total number of requests/tasks evaluated.
    latency_p50_ms:
        Median (50th percentile) response latency in milliseconds.
    latency_p99_ms:
        99th-percentile response latency in milliseconds.
    cost_per_task_usd:
        Average cost in USD per completed task.
    cost_budget_usd:
        Acceptable cost ceiling per task in USD.
    latency_target_p50_ms:
        Acceptable p50 latency target in milliseconds.
    latency_target_p99_ms:
        Acceptable p99 latency target in milliseconds.
    """

    uptime_fraction: float
    error_rate: float
    violation_count: int
    sample_size: int
    latency_p50_ms: float
    latency_p99_ms: float
    cost_per_task_usd: float
    cost_budget_usd: float
    latency_target_p50_ms: float = 500.0
    latency_target_p99_ms: float = 2000.0


# ---------------------------------------------------------------------------
# Dimension result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single readiness dimension.

    Attributes
    ----------
    name:
        Dimension identifier (e.g. "reliability").
    score:
        Score in [0.0, 100.0].
    weight:
        Weight of this dimension in the composite score.
    weighted_contribution:
        ``score * weight`` — the contribution to the composite.
    details:
        Human-readable explanation.
    """

    name: str
    score: float
    weight: float
    weighted_contribution: float
    details: str


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadinessReport:
    """Full production readiness report.

    Attributes
    ----------
    composite_score:
        Weighted average score across all dimensions, in [0.0, 100.0].
    grade:
        Qualitative :class:`ReadinessGrade`.
    dimensions:
        Individual :class:`DimensionScore` objects.
    is_production_ready:
        True when composite_score >= 85.
    evaluated_at:
        UTC timestamp of report generation.
    recommendations:
        List of actionable recommendations for improvement.
    """

    composite_score: float
    grade: ReadinessGrade
    dimensions: list[DimensionScore]
    is_production_ready: bool
    evaluated_at: datetime.datetime
    recommendations: list[str]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


# Dimension weights — must sum to 1.0
_WEIGHTS: dict[str, float] = {
    "reliability": 0.30,
    "safety": 0.30,
    "performance": 0.25,
    "cost": 0.15,
}


class ProductionReadinessScorer:
    """Compute composite production readiness scores.

    Example
    -------
    ::

        metrics = ReadinessInput(
            uptime_fraction=0.995,
            error_rate=0.01,
            violation_count=0,
            sample_size=1000,
            latency_p50_ms=120.0,
            latency_p99_ms=800.0,
            cost_per_task_usd=0.002,
            cost_budget_usd=0.01,
        )
        scorer = ProductionReadinessScorer()
        report = scorer.score(metrics)
        print(report.composite_score, report.grade)
    """

    def score(self, metrics: ReadinessInput) -> ReadinessReport:
        """Compute a :class:`ReadinessReport` from raw metrics.

        Parameters
        ----------
        metrics:
            The :class:`ReadinessInput` containing raw measurements.

        Returns
        -------
        ReadinessReport
            The full readiness report.
        """
        dimensions: list[DimensionScore] = [
            self._score_reliability(metrics),
            self._score_safety(metrics),
            self._score_performance(metrics),
            self._score_cost(metrics),
        ]

        composite = sum(d.weighted_contribution for d in dimensions)
        composite = round(min(100.0, max(0.0, composite)), 2)
        grade = self._grade(composite)
        recommendations = self._build_recommendations(dimensions, metrics)

        return ReadinessReport(
            composite_score=composite,
            grade=grade,
            dimensions=dimensions,
            is_production_ready=composite >= 85.0,
            evaluated_at=datetime.datetime.now(datetime.timezone.utc),
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_reliability(m: ReadinessInput) -> DimensionScore:
        """Score reliability from uptime and error rate."""
        uptime_score = m.uptime_fraction * 100.0
        error_penalty = m.error_rate * 100.0
        raw = max(0.0, uptime_score - error_penalty)
        raw = min(100.0, raw)
        weight = _WEIGHTS["reliability"]
        details = (
            f"Uptime {m.uptime_fraction:.1%}, error rate {m.error_rate:.1%}. "
            f"Raw score: {raw:.1f}"
        )
        return DimensionScore(
            name="reliability",
            score=raw,
            weight=weight,
            weighted_contribution=raw * weight,
            details=details,
        )

    @staticmethod
    def _score_safety(m: ReadinessInput) -> DimensionScore:
        """Score safety from violation rate."""
        if m.sample_size <= 0:
            raw = 0.0
        else:
            violation_rate = m.violation_count / m.sample_size
            # 0 violations → 100; 5%+ violations → 0
            raw = max(0.0, 100.0 - violation_rate * 2000.0)
        raw = min(100.0, raw)
        weight = _WEIGHTS["safety"]
        details = (
            f"{m.violation_count} violations in {m.sample_size} samples "
            f"({m.violation_count / max(1, m.sample_size):.2%} rate). "
            f"Raw score: {raw:.1f}"
        )
        return DimensionScore(
            name="safety",
            score=raw,
            weight=weight,
            weighted_contribution=raw * weight,
            details=details,
        )

    @staticmethod
    def _score_performance(m: ReadinessInput) -> DimensionScore:
        """Score performance from latency p50 and p99 vs targets."""
        target_p50 = max(1.0, m.latency_target_p50_ms)
        target_p99 = max(1.0, m.latency_target_p99_ms)

        p50_ratio = m.latency_p50_ms / target_p50
        p99_ratio = m.latency_p99_ms / target_p99

        # Score each: 100 at or below target, declining as ratio increases
        p50_score = max(0.0, 100.0 - (p50_ratio - 1.0) * 50.0) if p50_ratio > 1.0 else 100.0
        p99_score = max(0.0, 100.0 - (p99_ratio - 1.0) * 50.0) if p99_ratio > 1.0 else 100.0

        raw = (p50_score * 0.4 + p99_score * 0.6)
        raw = min(100.0, max(0.0, raw))
        weight = _WEIGHTS["performance"]
        details = (
            f"p50={m.latency_p50_ms:.0f}ms (target={target_p50:.0f}ms), "
            f"p99={m.latency_p99_ms:.0f}ms (target={target_p99:.0f}ms). "
            f"Raw score: {raw:.1f}"
        )
        return DimensionScore(
            name="performance",
            score=raw,
            weight=weight,
            weighted_contribution=raw * weight,
            details=details,
        )

    @staticmethod
    def _score_cost(m: ReadinessInput) -> DimensionScore:
        """Score cost efficiency against budget."""
        if m.cost_budget_usd <= 0.0:
            raw = 0.0
        else:
            ratio = m.cost_per_task_usd / m.cost_budget_usd
            if ratio <= 1.0:
                raw = 100.0
            else:
                # Linear decay: 2x budget → 0
                raw = max(0.0, 100.0 - (ratio - 1.0) * 100.0)
        raw = min(100.0, raw)
        weight = _WEIGHTS["cost"]
        details = (
            f"Cost per task ${m.cost_per_task_usd:.4f} vs budget ${m.cost_budget_usd:.4f} "
            f"(ratio={m.cost_per_task_usd / max(1e-9, m.cost_budget_usd):.2f}). "
            f"Raw score: {raw:.1f}"
        )
        return DimensionScore(
            name="cost",
            score=raw,
            weight=weight,
            weighted_contribution=raw * weight,
            details=details,
        )

    # ------------------------------------------------------------------
    # Grading and recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _grade(composite: float) -> ReadinessGrade:
        """Map a composite score to a :class:`ReadinessGrade`."""
        if composite >= 85.0:
            return ReadinessGrade.PRODUCTION_READY
        if composite >= 70.0:
            return ReadinessGrade.MOSTLY_READY
        if composite >= 50.0:
            return ReadinessGrade.NEEDS_WORK
        return ReadinessGrade.NOT_READY

    @staticmethod
    def _build_recommendations(
        dimensions: list[DimensionScore], m: ReadinessInput
    ) -> list[str]:
        """Build actionable recommendations based on dimension scores."""
        recs: list[str] = []
        for dim in dimensions:
            if dim.name == "reliability" and dim.score < 80.0:
                recs.append(
                    f"Improve reliability: uptime={m.uptime_fraction:.1%}, "
                    f"error_rate={m.error_rate:.1%}. Target uptime>99.5%, error_rate<1%."
                )
            if dim.name == "safety" and dim.score < 80.0:
                recs.append(
                    f"Reduce safety violations: {m.violation_count} in {m.sample_size} samples. "
                    "Review safety constraints and add guardrails."
                )
            if dim.name == "performance" and dim.score < 80.0:
                recs.append(
                    f"Improve latency: p50={m.latency_p50_ms:.0f}ms, "
                    f"p99={m.latency_p99_ms:.0f}ms. "
                    "Consider caching, model quantisation, or batching."
                )
            if dim.name == "cost" and dim.score < 80.0:
                recs.append(
                    f"Reduce cost: ${m.cost_per_task_usd:.4f}/task exceeds "
                    f"budget ${m.cost_budget_usd:.4f}. "
                    "Optimise prompts, use smaller models, or increase batching."
                )
        return recs


__all__ = [
    "DimensionScore",
    "ProductionReadinessScorer",
    "ReadinessGrade",
    "ReadinessInput",
    "ReadinessReport",
]
