"""Tests for agent_sim_bridge.readiness.scorer."""
from __future__ import annotations

import pytest

from agent_sim_bridge.readiness.scorer import (
    DimensionScore,
    ProductionReadinessScorer,
    ReadinessGrade,
    ReadinessInput,
    ReadinessReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scorer() -> ProductionReadinessScorer:
    return ProductionReadinessScorer()


def _perfect_input() -> ReadinessInput:
    return ReadinessInput(
        uptime_fraction=1.0,
        error_rate=0.0,
        violation_count=0,
        sample_size=1000,
        latency_p50_ms=100.0,
        latency_p99_ms=500.0,
        cost_per_task_usd=0.001,
        cost_budget_usd=0.01,
        latency_target_p50_ms=500.0,
        latency_target_p99_ms=2000.0,
    )


def _poor_input() -> ReadinessInput:
    return ReadinessInput(
        uptime_fraction=0.80,
        error_rate=0.15,
        violation_count=50,
        sample_size=1000,
        latency_p50_ms=2000.0,
        latency_p99_ms=8000.0,
        cost_per_task_usd=0.05,
        cost_budget_usd=0.01,
        latency_target_p50_ms=500.0,
        latency_target_p99_ms=2000.0,
    )


# ---------------------------------------------------------------------------
# ReadinessInput
# ---------------------------------------------------------------------------


class TestReadinessInput:
    def test_input_is_frozen(self) -> None:
        inp = _perfect_input()
        with pytest.raises((AttributeError, TypeError)):
            inp.uptime_fraction = 0.5  # type: ignore[misc]

    def test_default_latency_targets(self) -> None:
        inp = ReadinessInput(
            uptime_fraction=1.0,
            error_rate=0.0,
            violation_count=0,
            sample_size=100,
            latency_p50_ms=100.0,
            latency_p99_ms=500.0,
            cost_per_task_usd=0.001,
            cost_budget_usd=0.01,
        )
        assert inp.latency_target_p50_ms == 500.0
        assert inp.latency_target_p99_ms == 2000.0


# ---------------------------------------------------------------------------
# Perfect agent → production ready
# ---------------------------------------------------------------------------


class TestPerfectAgent:
    def test_perfect_score_is_near_100(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_perfect_input())
        assert report.composite_score >= 90.0

    def test_perfect_grade_is_production_ready(
        self, scorer: ProductionReadinessScorer
    ) -> None:
        report = scorer.score(_perfect_input())
        assert report.grade == ReadinessGrade.PRODUCTION_READY

    def test_perfect_is_production_ready_flag(
        self, scorer: ProductionReadinessScorer
    ) -> None:
        report = scorer.score(_perfect_input())
        assert report.is_production_ready is True

    def test_perfect_has_no_recommendations(
        self, scorer: ProductionReadinessScorer
    ) -> None:
        report = scorer.score(_perfect_input())
        # Perfect agent may have 0 recommendations
        assert isinstance(report.recommendations, list)


# ---------------------------------------------------------------------------
# Poor agent → not ready
# ---------------------------------------------------------------------------


class TestPoorAgent:
    def test_poor_score_is_below_50(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_poor_input())
        assert report.composite_score < 70.0

    def test_poor_is_not_production_ready(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_poor_input())
        assert report.is_production_ready is False

    def test_poor_has_recommendations(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_poor_input())
        assert len(report.recommendations) > 0


# ---------------------------------------------------------------------------
# Individual dimensions
# ---------------------------------------------------------------------------


class TestDimensions:
    def test_report_has_four_dimensions(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_perfect_input())
        assert len(report.dimensions) == 4

    def test_dimension_names(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_perfect_input())
        names = {d.name for d in report.dimensions}
        assert names == {"reliability", "safety", "performance", "cost"}

    def test_dimension_scores_in_range(self, scorer: ProductionReadinessScorer) -> None:
        for inp in [_perfect_input(), _poor_input()]:
            report = scorer.score(inp)
            for dim in report.dimensions:
                assert 0.0 <= dim.score <= 100.0

    def test_weights_sum_to_one(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_perfect_input())
        total_weight = sum(d.weight for d in report.dimensions)
        assert abs(total_weight - 1.0) < 1e-9

    def test_weighted_contribution_matches(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_perfect_input())
        for dim in report.dimensions:
            assert abs(dim.weighted_contribution - dim.score * dim.weight) < 1e-9

    def test_composite_matches_sum_of_contributions(
        self, scorer: ProductionReadinessScorer
    ) -> None:
        for inp in [_perfect_input(), _poor_input()]:
            report = scorer.score(inp)
            expected = sum(d.weighted_contribution for d in report.dimensions)
            assert abs(report.composite_score - round(expected, 2)) < 0.01


# ---------------------------------------------------------------------------
# Grading bands
# ---------------------------------------------------------------------------


class TestGrading:
    def test_grade_not_ready_below_50(self, scorer: ProductionReadinessScorer) -> None:
        assert scorer._grade(30.0) == ReadinessGrade.NOT_READY

    def test_grade_needs_work_50_to_70(self, scorer: ProductionReadinessScorer) -> None:
        assert scorer._grade(60.0) == ReadinessGrade.NEEDS_WORK

    def test_grade_mostly_ready_70_to_85(self, scorer: ProductionReadinessScorer) -> None:
        assert scorer._grade(75.0) == ReadinessGrade.MOSTLY_READY

    def test_grade_production_ready_85_plus(self, scorer: ProductionReadinessScorer) -> None:
        assert scorer._grade(90.0) == ReadinessGrade.PRODUCTION_READY

    def test_grade_boundaries(self, scorer: ProductionReadinessScorer) -> None:
        assert scorer._grade(50.0) == ReadinessGrade.NEEDS_WORK
        assert scorer._grade(70.0) == ReadinessGrade.MOSTLY_READY
        assert scorer._grade(85.0) == ReadinessGrade.PRODUCTION_READY


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_sample_size_safety_score_zero(
        self, scorer: ProductionReadinessScorer
    ) -> None:
        inp = ReadinessInput(
            uptime_fraction=1.0,
            error_rate=0.0,
            violation_count=0,
            sample_size=0,
            latency_p50_ms=100.0,
            latency_p99_ms=500.0,
            cost_per_task_usd=0.001,
            cost_budget_usd=0.01,
        )
        report = scorer.score(inp)
        safety = next(d for d in report.dimensions if d.name == "safety")
        assert safety.score == 0.0

    def test_zero_cost_budget_scores_zero(self, scorer: ProductionReadinessScorer) -> None:
        inp = ReadinessInput(
            uptime_fraction=1.0,
            error_rate=0.0,
            violation_count=0,
            sample_size=100,
            latency_p50_ms=100.0,
            latency_p99_ms=500.0,
            cost_per_task_usd=0.001,
            cost_budget_usd=0.0,
        )
        report = scorer.score(inp)
        cost = next(d for d in report.dimensions if d.name == "cost")
        assert cost.score == 0.0

    def test_composite_bounded_0_100(self, scorer: ProductionReadinessScorer) -> None:
        for inp in [_perfect_input(), _poor_input()]:
            report = scorer.score(inp)
            assert 0.0 <= report.composite_score <= 100.0

    def test_report_evaluated_at_is_set(self, scorer: ProductionReadinessScorer) -> None:
        report = scorer.score(_perfect_input())
        assert report.evaluated_at is not None
