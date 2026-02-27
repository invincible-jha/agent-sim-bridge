"""Unit tests for the sim-to-real gap estimation engine.

Covers:
- statistics.py — kl_divergence, wasserstein_distance_1d, maximum_mean_discrepancy,
  jensen_shannon_divergence, normalize_distribution, descriptive_stats
- estimator.py — GapDimension, GapMetric, DimensionGap, GapEstimator
- report.py — GapReport, GapReporter (text/json/markdown formats)
- cli gap commands — gap estimate, gap compare
- Edge cases: empty inputs, single-element, skewed distributions, all-zeros, etc.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_sim_bridge.cli.main import cli
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform(n: int) -> list[float]:
    """Uniform distribution over n bins."""
    return [1.0 / n] * n


def _runner() -> CliRunner:
    return CliRunner()


def _write_gap_data(directory: str, sim_dims: dict, real_dims: dict) -> tuple[str, str]:
    """Write sim and real JSON files and return (sim_path, real_path)."""
    sim_path = Path(directory) / "sim.json"
    real_path = Path(directory) / "real.json"
    sim_path.write_text(
        json.dumps({"dimensions": sim_dims}), encoding="utf-8"
    )
    real_path.write_text(
        json.dumps({"dimensions": real_dims}), encoding="utf-8"
    )
    return str(sim_path), str(real_path)


def _dim_entry(values: list[float]) -> dict:
    return {"sim": values, "real": values}


# ---------------------------------------------------------------------------
# normalize_distribution
# ---------------------------------------------------------------------------


class TestNormalizeDistribution:
    def test_uniform_stays_uniform(self) -> None:
        result = normalize_distribution([1.0, 1.0, 1.0, 1.0])
        assert result == pytest.approx([0.25, 0.25, 0.25, 0.25])

    def test_counts_normalised_correctly(self) -> None:
        result = normalize_distribution([1.0, 2.0, 1.0])
        assert result == pytest.approx([0.25, 0.5, 0.25])

    def test_sum_is_one(self) -> None:
        result = normalize_distribution([3.0, 7.0, 10.0])
        assert sum(result) == pytest.approx(1.0)

    def test_single_element(self) -> None:
        result = normalize_distribution([42.0])
        assert result == pytest.approx([1.0])

    def test_already_normalised_unchanged(self) -> None:
        distribution = [0.2, 0.3, 0.5]
        result = normalize_distribution(distribution)
        assert result == pytest.approx(distribution)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            normalize_distribution([])

    def test_all_zeros_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_distribution([0.0, 0.0, 0.0])

    def test_preserves_relative_proportions(self) -> None:
        result = normalize_distribution([1.0, 3.0])
        assert result[1] == pytest.approx(3 * result[0])

    def test_large_values_normalised(self) -> None:
        result = normalize_distribution([1e6, 1e6])
        assert result == pytest.approx([0.5, 0.5])

    def test_very_small_values(self) -> None:
        result = normalize_distribution([1e-10, 1e-10])
        assert result == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------


class TestKlDivergence:
    def test_identical_distributions_returns_zero(self) -> None:
        dist = [0.25, 0.25, 0.25, 0.25]
        assert kl_divergence(dist, dist) == pytest.approx(0.0, abs=1e-7)

    def test_known_value_two_bins(self) -> None:
        # P=[0.9,0.1], Q=[0.1,0.9] — large KL expected
        kl = kl_divergence([0.9, 0.1], [0.1, 0.9])
        assert kl > 1.0

    def test_result_is_non_negative(self) -> None:
        kl = kl_divergence([0.4, 0.6], [0.3, 0.7])
        assert kl >= 0.0

    def test_not_symmetric(self) -> None:
        # Use three bins with clearly different values so P||Q != Q||P
        p = [0.7, 0.2, 0.1]
        q = [0.1, 0.1, 0.8]
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        # Both are positive but generally differ — verify neither is zero
        # and that they are not the same (within a strict tolerance).
        assert kl_pq > 0.0
        assert kl_qp > 0.0
        # KL divergence is not symmetric in general
        assert abs(kl_pq - kl_qp) > 1e-4

    def test_empty_p_raises(self) -> None:
        with pytest.raises(ValueError):
            kl_divergence([], [1.0])

    def test_empty_q_raises(self) -> None:
        with pytest.raises(ValueError):
            kl_divergence([1.0], [])

    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_zero_values_handled_with_smoothing(self) -> None:
        # Should not raise; epsilon smoothing handles the zeros.
        result = kl_divergence([0.0, 1.0], [1.0, 0.0])
        assert result >= 0.0
        assert math.isfinite(result)

    def test_uniform_vs_peaked_positive(self) -> None:
        uniform = [0.25, 0.25, 0.25, 0.25]
        peaked = [0.97, 0.01, 0.01, 0.01]
        assert kl_divergence(peaked, uniform) > 0.0

    def test_single_bin_identical(self) -> None:
        assert kl_divergence([1.0], [1.0]) == pytest.approx(0.0, abs=1e-7)

    def test_all_zeros_both_distributions(self) -> None:
        # Both zero → after smoothing they are identical → KL ≈ 0
        result = kl_divergence([0.0, 0.0], [0.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_very_skewed_distribution(self) -> None:
        p = [0.999] + [0.001 / 9] * 9
        q = [1.0 / 10] * 10
        result = kl_divergence(p, q)
        assert result > 0.5

    def test_kl_increases_with_divergence(self) -> None:
        kl_close = kl_divergence([0.5, 0.5], [0.49, 0.51])
        kl_far = kl_divergence([0.5, 0.5], [0.1, 0.9])
        assert kl_far > kl_close


# ---------------------------------------------------------------------------
# wasserstein_distance_1d
# ---------------------------------------------------------------------------


class TestWassersteinDistance1d:
    def test_identical_distributions_returns_zero(self) -> None:
        dist = [0.25, 0.25, 0.25, 0.25]
        assert wasserstein_distance_1d(dist, dist) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution(self) -> None:
        # All mass in first bin vs all mass in last bin → large distance
        p = [1.0, 0.0, 0.0, 0.0]
        q = [0.0, 0.0, 0.0, 1.0]
        dist = wasserstein_distance_1d(p, q)
        assert dist > 0.0

    def test_result_is_non_negative(self) -> None:
        assert wasserstein_distance_1d([0.3, 0.7], [0.6, 0.4]) >= 0.0

    def test_symmetric(self) -> None:
        p = [0.1, 0.4, 0.4, 0.1]
        q = [0.4, 0.1, 0.1, 0.4]
        assert wasserstein_distance_1d(p, q) == pytest.approx(
            wasserstein_distance_1d(q, p), abs=1e-9
        )

    def test_empty_p_raises(self) -> None:
        with pytest.raises(ValueError):
            wasserstein_distance_1d([], [1.0])

    def test_empty_q_raises(self) -> None:
        with pytest.raises(ValueError):
            wasserstein_distance_1d([1.0], [])

    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            wasserstein_distance_1d([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_single_bin_identical(self) -> None:
        assert wasserstein_distance_1d([1.0], [1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_adjacent_bins(self) -> None:
        # Moving mass one bin to the right → non-zero distance
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        dist = wasserstein_distance_1d(p, q)
        assert dist > 0.0

    def test_zero_values_handled(self) -> None:
        result = wasserstein_distance_1d([0.0, 1.0], [1.0, 0.0])
        assert math.isfinite(result)
        assert result >= 0.0

    def test_larger_distance_for_more_separated_distributions(self) -> None:
        close = wasserstein_distance_1d([0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0])
        far = wasserstein_distance_1d([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        assert far > close

    def test_all_zeros_both(self) -> None:
        # After smoothing, identical → distance ≈ 0
        result = wasserstein_distance_1d([0.0, 0.0], [0.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# maximum_mean_discrepancy
# ---------------------------------------------------------------------------


class TestMaximumMeanDiscrepancy:
    def test_same_distribution_near_zero(self) -> None:
        samples = [0.0, 1.0, 2.0, 3.0, 4.0]
        mmd = maximum_mean_discrepancy(samples, samples)
        assert mmd == pytest.approx(0.0, abs=1e-9)

    def test_different_distributions_positive(self) -> None:
        x = [0.0, 0.1, 0.2, 0.3]
        y = [5.0, 5.1, 5.2, 5.3]
        mmd = maximum_mean_discrepancy(x, y)
        assert mmd > 0.0

    def test_result_is_non_negative(self) -> None:
        mmd = maximum_mean_discrepancy([1.0, 2.0], [1.5, 2.5])
        assert mmd >= 0.0

    def test_symmetric(self) -> None:
        x = [0.0, 1.0, 2.0]
        y = [3.0, 4.0, 5.0]
        assert maximum_mean_discrepancy(x, y) == pytest.approx(
            maximum_mean_discrepancy(y, x), abs=1e-9
        )

    def test_empty_x_raises(self) -> None:
        with pytest.raises(ValueError):
            maximum_mean_discrepancy([], [1.0])

    def test_empty_y_raises(self) -> None:
        with pytest.raises(ValueError):
            maximum_mean_discrepancy([1.0], [])

    def test_zero_bandwidth_raises(self) -> None:
        with pytest.raises(ValueError, match="bandwidth"):
            maximum_mean_discrepancy([1.0], [1.0], bandwidth=0.0)

    def test_negative_bandwidth_raises(self) -> None:
        with pytest.raises(ValueError, match="bandwidth"):
            maximum_mean_discrepancy([1.0], [1.0], bandwidth=-1.0)

    def test_bandwidth_affects_sensitivity(self) -> None:
        x = [0.0, 1.0]
        y = [2.0, 3.0]
        mmd_narrow = maximum_mean_discrepancy(x, y, bandwidth=0.1)
        mmd_wide = maximum_mean_discrepancy(x, y, bandwidth=100.0)
        # With very wide bandwidth, kernel values are all nearly 1 → MMD → 0
        assert mmd_wide < mmd_narrow

    def test_single_element_same(self) -> None:
        assert maximum_mean_discrepancy([1.0], [1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_single_element_different(self) -> None:
        mmd = maximum_mean_discrepancy([0.0], [100.0])
        assert mmd > 0.0

    def test_large_separation_high_mmd(self) -> None:
        x = list(range(10))
        y = [v + 1000 for v in range(10)]
        mmd = maximum_mean_discrepancy(x, y, bandwidth=1.0)
        # With bandwidth=1 and separation=1000, cross-kernel values ≈ 0 so
        # MMD² ≈ E[k(x,x)] + E[k(y,y)] - 2*0 = two non-zero terms → mmd > 0.5
        assert mmd > 0.5

    def test_mmd_increases_with_distribution_separation(self) -> None:
        x = [0.0, 1.0, 2.0, 3.0]
        y_close = [0.5, 1.5, 2.5, 3.5]
        y_far = [10.0, 11.0, 12.0, 13.0]
        mmd_close = maximum_mean_discrepancy(x, y_close)
        mmd_far = maximum_mean_discrepancy(x, y_far)
        assert mmd_far > mmd_close


# ---------------------------------------------------------------------------
# jensen_shannon_divergence
# ---------------------------------------------------------------------------


class TestJensenShannonDivergence:
    def test_identical_distributions_returns_zero(self) -> None:
        dist = [0.25, 0.25, 0.25, 0.25]
        assert jensen_shannon_divergence(dist, dist) == pytest.approx(0.0, abs=1e-7)

    def test_symmetric(self) -> None:
        p = [0.7, 0.3]
        q = [0.3, 0.7]
        assert jensen_shannon_divergence(p, q) == pytest.approx(
            jensen_shannon_divergence(q, p), abs=1e-9
        )

    def test_bounded_above_by_ln2(self) -> None:
        # Maximum JSD = ln(2) ≈ 0.693 for completely disjoint distributions
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        jsd = jensen_shannon_divergence(p, q)
        assert jsd <= math.log(2) + 1e-6

    def test_result_non_negative(self) -> None:
        assert jensen_shannon_divergence([0.4, 0.6], [0.6, 0.4]) >= 0.0

    def test_empty_p_raises(self) -> None:
        with pytest.raises(ValueError):
            jensen_shannon_divergence([], [1.0])

    def test_empty_q_raises(self) -> None:
        with pytest.raises(ValueError):
            jensen_shannon_divergence([1.0], [])

    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            jensen_shannon_divergence([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_disjoint_distributions_near_ln2(self) -> None:
        # With epsilon smoothing the limit is not exactly ln(2) but approaches it
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        jsd = jensen_shannon_divergence(p, q)
        assert jsd > 0.5

    def test_zero_values_handled(self) -> None:
        result = jensen_shannon_divergence([0.0, 1.0], [1.0, 0.0])
        assert math.isfinite(result)
        assert result >= 0.0

    def test_single_bin_identical(self) -> None:
        assert jensen_shannon_divergence([1.0], [1.0]) == pytest.approx(0.0, abs=1e-7)

    def test_increases_with_divergence(self) -> None:
        jsd_close = jensen_shannon_divergence([0.5, 0.5], [0.49, 0.51])
        jsd_far = jensen_shannon_divergence([0.5, 0.5], [0.1, 0.9])
        assert jsd_far > jsd_close

    def test_uniform_vs_peaked(self) -> None:
        uniform = [0.25, 0.25, 0.25, 0.25]
        peaked = [0.97, 0.01, 0.01, 0.01]
        jsd = jensen_shan_divergence = jensen_shannon_divergence(peaked, uniform)
        assert jsd > 0.0

    def test_all_zeros_both_distributions(self) -> None:
        # After smoothing → identical → JSD ≈ 0
        result = jensen_shannon_divergence([0.0, 0.0], [0.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# descriptive_stats
# ---------------------------------------------------------------------------


class TestDescriptiveStats:
    def test_basic_five_elements(self) -> None:
        stats = descriptive_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_median_odd_count(self) -> None:
        stats = descriptive_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["median"] == pytest.approx(3.0)

    def test_median_even_count(self) -> None:
        stats = descriptive_stats([1.0, 2.0, 3.0, 4.0])
        assert stats["median"] == pytest.approx(2.5)

    def test_q25_q75_present(self) -> None:
        stats = descriptive_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert "q25" in stats
        assert "q75" in stats

    def test_q25_less_than_median(self) -> None:
        stats = descriptive_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["q25"] < stats["median"]

    def test_q75_greater_than_median(self) -> None:
        stats = descriptive_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["q75"] > stats["median"]

    def test_std_zero_for_constant_values(self) -> None:
        stats = descriptive_stats([7.0, 7.0, 7.0])
        assert stats["std"] == pytest.approx(0.0)

    def test_std_positive_for_spread_values(self) -> None:
        stats = descriptive_stats([0.0, 10.0])
        assert stats["std"] > 0.0

    def test_single_element(self) -> None:
        stats = descriptive_stats([42.0])
        assert stats["mean"] == pytest.approx(42.0)
        assert stats["min"] == pytest.approx(42.0)
        assert stats["max"] == pytest.approx(42.0)
        assert stats["std"] == pytest.approx(0.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            descriptive_stats([])

    def test_negative_values(self) -> None:
        stats = descriptive_stats([-5.0, -3.0, -1.0])
        assert stats["mean"] == pytest.approx(-3.0)
        assert stats["min"] == pytest.approx(-5.0)

    def test_all_keys_present(self) -> None:
        stats = descriptive_stats([1.0, 2.0])
        for key in ("mean", "std", "min", "max", "median", "q25", "q75"):
            assert key in stats

    def test_unsorted_input_handled(self) -> None:
        stats = descriptive_stats([5.0, 1.0, 3.0, 2.0, 4.0])
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# GapDimension
# ---------------------------------------------------------------------------


class TestGapDimension:
    def test_creation(self) -> None:
        dim = GapDimension(
            name="torque",
            sim_distribution=[0.1, 0.3, 0.4, 0.2],
            real_distribution=[0.15, 0.25, 0.35, 0.25],
        )
        assert dim.name == "torque"
        assert len(dim.sim_distribution) == 4

    def test_frozen(self) -> None:
        dim = GapDimension(name="x", sim_distribution=[1.0], real_distribution=[1.0])
        with pytest.raises((AttributeError, TypeError)):
            dim.name = "y"  # type: ignore[misc]

    def test_equality(self) -> None:
        dim_a = GapDimension(name="x", sim_distribution=[1.0], real_distribution=[1.0])
        dim_b = GapDimension(name="x", sim_distribution=[1.0], real_distribution=[1.0])
        assert dim_a == dim_b


# ---------------------------------------------------------------------------
# GapMetric
# ---------------------------------------------------------------------------


class TestGapMetric:
    def test_all_four_members(self) -> None:
        assert len(list(GapMetric)) == 4

    def test_member_values(self) -> None:
        assert GapMetric.KL_DIVERGENCE.value == "KL_DIVERGENCE"
        assert GapMetric.WASSERSTEIN.value == "WASSERSTEIN"
        assert GapMetric.MMD.value == "MMD"
        assert GapMetric.JENSEN_SHANNON.value == "JENSEN_SHANNON"


# ---------------------------------------------------------------------------
# DimensionGap
# ---------------------------------------------------------------------------


class TestDimensionGap:
    def test_creation(self) -> None:
        gap = DimensionGap(
            dimension_name="torque",
            metric=GapMetric.KL_DIVERGENCE,
            value=0.05,
            interpretation="low",
        )
        assert gap.dimension_name == "torque"
        assert gap.value == pytest.approx(0.05)

    def test_frozen(self) -> None:
        gap = DimensionGap(
            dimension_name="x",
            metric=GapMetric.WASSERSTEIN,
            value=0.1,
            interpretation="medium",
        )
        with pytest.raises((AttributeError, TypeError)):
            gap.value = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GapEstimator — single dimension
# ---------------------------------------------------------------------------


class TestGapEstimatorSingleDimension:
    def setup_method(self) -> None:
        self.estimator = GapEstimator()
        self.identical_dim = GapDimension(
            name="identical",
            sim_distribution=[0.25, 0.25, 0.25, 0.25],
            real_distribution=[0.25, 0.25, 0.25, 0.25],
        )
        self.shifted_dim = GapDimension(
            name="shifted",
            sim_distribution=[0.7, 0.2, 0.05, 0.05],
            real_distribution=[0.05, 0.05, 0.2, 0.7],
        )

    def test_returns_one_result_per_metric(self) -> None:
        results = self.estimator.estimate_dimension(self.identical_dim)
        assert len(results) == 4

    def test_identical_distributions_low_values(self) -> None:
        results = self.estimator.estimate_dimension(self.identical_dim)
        for result in results:
            assert result.value == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distributions_positive_values(self) -> None:
        results = self.estimator.estimate_dimension(self.shifted_dim)
        for result in results:
            assert result.value > 0.0

    def test_dimension_name_preserved(self) -> None:
        results = self.estimator.estimate_dimension(self.identical_dim)
        for result in results:
            assert result.dimension_name == "identical"

    def test_interpretation_is_valid(self) -> None:
        results = self.estimator.estimate_dimension(self.shifted_dim)
        for result in results:
            assert result.interpretation in ("low", "medium", "high")

    def test_custom_metrics_subset(self) -> None:
        estimator = GapEstimator(metrics=[GapMetric.KL_DIVERGENCE])
        results = estimator.estimate_dimension(self.identical_dim)
        assert len(results) == 1
        assert results[0].metric == GapMetric.KL_DIVERGENCE

    def test_single_metric_wasserstein(self) -> None:
        estimator = GapEstimator(metrics=[GapMetric.WASSERSTEIN])
        results = estimator.estimate_dimension(self.shifted_dim)
        assert len(results) == 1
        assert results[0].metric == GapMetric.WASSERSTEIN

    def test_interpretation_low_for_identical(self) -> None:
        results = self.estimator.estimate_dimension(self.identical_dim)
        for result in results:
            assert result.interpretation == "low"

    def test_interpretation_high_for_maximally_shifted(self) -> None:
        dim = GapDimension(
            name="extreme",
            sim_distribution=[1.0, 0.0, 0.0, 0.0],
            real_distribution=[0.0, 0.0, 0.0, 1.0],
        )
        results = self.estimator.estimate_dimension(dim)
        high_count = sum(1 for r in results if r.interpretation == "high")
        assert high_count >= 1


# ---------------------------------------------------------------------------
# GapEstimator — multiple dimensions
# ---------------------------------------------------------------------------


class TestGapEstimatorMultipleDimensions:
    def setup_method(self) -> None:
        self.estimator = GapEstimator()
        self.dimensions = [
            GapDimension(
                name="dim_a",
                sim_distribution=[0.25, 0.25, 0.25, 0.25],
                real_distribution=[0.25, 0.25, 0.25, 0.25],
            ),
            GapDimension(
                name="dim_b",
                sim_distribution=[0.8, 0.1, 0.05, 0.05],
                real_distribution=[0.05, 0.05, 0.1, 0.8],
            ),
        ]

    def test_returns_all_dimensions(self) -> None:
        results = self.estimator.estimate_all(self.dimensions)
        assert "dim_a" in results
        assert "dim_b" in results

    def test_each_dimension_has_correct_results_count(self) -> None:
        results = self.estimator.estimate_all(self.dimensions)
        assert len(results["dim_a"]) == 4
        assert len(results["dim_b"]) == 4

    def test_empty_dimensions_list(self) -> None:
        results = self.estimator.estimate_all([])
        assert results == {}

    def test_single_dimension(self) -> None:
        dim = GapDimension(name="solo", sim_distribution=[0.5, 0.5], real_distribution=[0.5, 0.5])
        results = self.estimator.estimate_all([dim])
        assert "solo" in results

    def test_dim_a_lower_gap_than_dim_b(self) -> None:
        results = self.estimator.estimate_all(self.dimensions)
        kl_a = next(r.value for r in results["dim_a"] if r.metric == GapMetric.KL_DIVERGENCE)
        kl_b = next(r.value for r in results["dim_b"] if r.metric == GapMetric.KL_DIVERGENCE)
        assert kl_b > kl_a

    def test_repr(self) -> None:
        assert "GapEstimator" in repr(self.estimator)


# ---------------------------------------------------------------------------
# GapEstimator — overall_gap_score
# ---------------------------------------------------------------------------


class TestOverallGapScore:
    def setup_method(self) -> None:
        self.estimator = GapEstimator()

    def test_identical_distributions_score_near_zero(self) -> None:
        dim = GapDimension(
            name="x",
            sim_distribution=[0.25, 0.25, 0.25, 0.25],
            real_distribution=[0.25, 0.25, 0.25, 0.25],
        )
        results = self.estimator.estimate_all([dim])
        score = self.estimator.overall_gap_score(results)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_empty_results_returns_zero(self) -> None:
        score = self.estimator.overall_gap_score({})
        assert score == pytest.approx(0.0)

    def test_score_between_zero_and_one(self) -> None:
        dim = GapDimension(
            name="x",
            sim_distribution=[0.8, 0.1, 0.1],
            real_distribution=[0.1, 0.1, 0.8],
        )
        results = self.estimator.estimate_all([dim])
        score = self.estimator.overall_gap_score(results)
        assert 0.0 <= score <= 1.0

    def test_high_gap_score_higher_than_low_gap(self) -> None:
        low_dim = GapDimension(
            name="low",
            sim_distribution=[0.25, 0.25, 0.25, 0.25],
            real_distribution=[0.25, 0.25, 0.25, 0.25],
        )
        high_dim = GapDimension(
            name="high",
            sim_distribution=[1.0, 0.0, 0.0, 0.0],
            real_distribution=[0.0, 0.0, 0.0, 1.0],
        )
        low_results = self.estimator.estimate_all([low_dim])
        high_results = self.estimator.estimate_all([high_dim])
        assert self.estimator.overall_gap_score(high_results) > self.estimator.overall_gap_score(
            low_results
        )

    def test_score_capped_at_one(self) -> None:
        dim = GapDimension(
            name="extreme",
            sim_distribution=[1.0, 0.0],
            real_distribution=[0.0, 1.0],
        )
        results = self.estimator.estimate_all([dim])
        score = self.estimator.overall_gap_score(results)
        assert score <= 1.0

    def test_multiple_dimensions_averaged(self) -> None:
        dims = [
            GapDimension(
                name=f"d{i}",
                sim_distribution=[0.25, 0.25, 0.25, 0.25],
                real_distribution=[0.25, 0.25, 0.25, 0.25],
            )
            for i in range(3)
        ]
        results = self.estimator.estimate_all(dims)
        score = self.estimator.overall_gap_score(results)
        assert score == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# GapReporter — report generation
# ---------------------------------------------------------------------------


class TestGapReporterGeneration:
    def setup_method(self) -> None:
        self.reporter = GapReporter()
        self.estimator = GapEstimator()
        self.dim = GapDimension(
            name="velocity",
            sim_distribution=[0.1, 0.3, 0.4, 0.2],
            real_distribution=[0.15, 0.25, 0.35, 0.25],
        )
        self.results = self.estimator.estimate_all([self.dim])

    def test_report_has_report_id(self) -> None:
        report = self.reporter.generate_report(self.results, [self.dim])
        assert len(report.report_id) > 0

    def test_report_id_is_unique(self) -> None:
        r1 = self.reporter.generate_report(self.results, [self.dim])
        r2 = self.reporter.generate_report(self.results, [self.dim])
        assert r1.report_id != r2.report_id

    def test_report_has_created_at(self) -> None:
        report = self.reporter.generate_report(self.results, [self.dim])
        assert report.created_at is not None

    def test_report_overall_score(self) -> None:
        report = self.reporter.generate_report(self.results, [self.dim])
        assert 0.0 <= report.overall_score <= 1.0

    def test_report_contains_dimensions(self) -> None:
        report = self.reporter.generate_report(self.results, [self.dim])
        assert "velocity" in report.dimensions

    def test_report_has_summary(self) -> None:
        report = self.reporter.generate_report(self.results, [self.dim])
        assert len(report.summary) > 0

    def test_report_has_recommendations(self) -> None:
        report = self.reporter.generate_report(self.results, [self.dim])
        assert isinstance(report.recommendations, list)

    def test_frozen_report(self) -> None:
        report = self.reporter.generate_report(self.results, [self.dim])
        with pytest.raises((AttributeError, TypeError)):
            report.overall_score = 0.5  # type: ignore[misc]

    def test_high_gap_generates_recommendations(self) -> None:
        extreme_dim = GapDimension(
            name="extreme",
            sim_distribution=[1.0, 0.0, 0.0, 0.0],
            real_distribution=[0.0, 0.0, 0.0, 1.0],
        )
        extreme_results = self.estimator.estimate_all([extreme_dim])
        report = self.reporter.generate_report(extreme_results, [extreme_dim])
        assert len(report.recommendations) >= 1

    def test_low_gap_positive_recommendation(self) -> None:
        uniform = [0.25, 0.25, 0.25, 0.25]
        low_dim = GapDimension(
            name="low",
            sim_distribution=uniform,
            real_distribution=uniform,
        )
        low_results = self.estimator.estimate_all([low_dim])
        report = self.reporter.generate_report(low_results, [low_dim])
        # Should include positive guidance
        all_recs = " ".join(report.recommendations).lower()
        assert "proceed" in all_recs or "low" in all_recs or "good" in all_recs

    def test_repr(self) -> None:
        assert "GapReporter" in repr(self.reporter)


# ---------------------------------------------------------------------------
# GapReporter — format_text
# ---------------------------------------------------------------------------


class TestGapReporterFormatText:
    def setup_method(self) -> None:
        self.reporter = GapReporter()
        estimator = GapEstimator()
        self.dim = GapDimension(
            name="position",
            sim_distribution=[0.2, 0.3, 0.3, 0.2],
            real_distribution=[0.25, 0.25, 0.25, 0.25],
        )
        results = estimator.estimate_all([self.dim])
        self.report = self.reporter.generate_report(results, [self.dim])

    def test_contains_report_id(self) -> None:
        text = self.reporter.format_text(self.report)
        assert self.report.report_id in text

    def test_contains_overall_score(self) -> None:
        text = self.reporter.format_text(self.report)
        assert "Overall Score" in text or "overall" in text.lower()

    def test_contains_dimension_name(self) -> None:
        text = self.reporter.format_text(self.report)
        assert "position" in text

    def test_contains_metric_names(self) -> None:
        text = self.reporter.format_text(self.report)
        assert "KL_DIVERGENCE" in text or "WASSERSTEIN" in text

    def test_contains_summary(self) -> None:
        text = self.reporter.format_text(self.report)
        assert self.report.summary in text

    def test_returns_string(self) -> None:
        assert isinstance(self.reporter.format_text(self.report), str)

    def test_has_separator_lines(self) -> None:
        text = self.reporter.format_text(self.report)
        assert "=" in text


# ---------------------------------------------------------------------------
# GapReporter — format_json
# ---------------------------------------------------------------------------


class TestGapReporterFormatJson:
    def setup_method(self) -> None:
        self.reporter = GapReporter()
        estimator = GapEstimator()
        dim = GapDimension(
            name="force",
            sim_distribution=[0.3, 0.4, 0.3],
            real_distribution=[0.33, 0.34, 0.33],
        )
        results = estimator.estimate_all([dim])
        self.report = self.reporter.generate_report(results, [dim])

    def test_valid_json(self) -> None:
        json_str = self.reporter.format_json(self.report)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_contains_report_id(self) -> None:
        parsed = json.loads(self.reporter.format_json(self.report))
        assert parsed["report_id"] == self.report.report_id

    def test_contains_overall_score(self) -> None:
        parsed = json.loads(self.reporter.format_json(self.report))
        assert "overall_score" in parsed

    def test_contains_dimensions(self) -> None:
        parsed = json.loads(self.reporter.format_json(self.report))
        assert "dimensions" in parsed
        assert "force" in parsed["dimensions"]

    def test_dimension_has_metric_value_interpretation(self) -> None:
        parsed = json.loads(self.reporter.format_json(self.report))
        for entry in parsed["dimensions"]["force"]:
            assert "metric" in entry
            assert "value" in entry
            assert "interpretation" in entry

    def test_contains_recommendations(self) -> None:
        parsed = json.loads(self.reporter.format_json(self.report))
        assert "recommendations" in parsed
        assert isinstance(parsed["recommendations"], list)

    def test_contains_created_at(self) -> None:
        parsed = json.loads(self.reporter.format_json(self.report))
        assert "created_at" in parsed

    def test_returns_string(self) -> None:
        assert isinstance(self.reporter.format_json(self.report), str)


# ---------------------------------------------------------------------------
# GapReporter — format_markdown
# ---------------------------------------------------------------------------


class TestGapReporterFormatMarkdown:
    def setup_method(self) -> None:
        self.reporter = GapReporter()
        estimator = GapEstimator()
        dim = GapDimension(
            name="torque",
            sim_distribution=[0.1, 0.4, 0.4, 0.1],
            real_distribution=[0.2, 0.3, 0.3, 0.2],
        )
        results = estimator.estimate_all([dim])
        self.report = self.reporter.generate_report(results, [dim])

    def test_has_h1_title(self) -> None:
        md = self.reporter.format_markdown(self.report)
        assert "# Sim-to-Real Gap Report" in md

    def test_contains_dimension_heading(self) -> None:
        md = self.reporter.format_markdown(self.report)
        assert "torque" in md

    def test_contains_table_header(self) -> None:
        md = self.reporter.format_markdown(self.report)
        assert "| Metric |" in md

    def test_contains_overall_score(self) -> None:
        md = self.reporter.format_markdown(self.report)
        assert "Overall Score" in md or "overall" in md.lower()

    def test_contains_recommendations_section(self) -> None:
        md = self.reporter.format_markdown(self.report)
        assert "## Recommendations" in md or "Recommendations" in md

    def test_returns_string(self) -> None:
        assert isinstance(self.reporter.format_markdown(self.report), str)

    def test_report_id_in_output(self) -> None:
        md = self.reporter.format_markdown(self.report)
        assert self.report.report_id in md


# ---------------------------------------------------------------------------
# GapReporter — recommendations heuristics
# ---------------------------------------------------------------------------


class TestGapReporterRecommendations:
    def setup_method(self) -> None:
        self.reporter = GapReporter()
        self.estimator = GapEstimator()

    def _run(self, sim: list[float], real: list[float]) -> list[str]:
        dim = GapDimension(name="d", sim_distribution=sim, real_distribution=real)
        results = self.estimator.estimate_all([dim])
        report = self.reporter.generate_report(results, [dim])
        return report.recommendations

    def test_high_gap_includes_domain_randomisation(self) -> None:
        recs = self._run([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        combined = " ".join(recs).lower()
        assert "domain randomis" in combined or "randomiz" in combined

    def test_high_gap_includes_calibration(self) -> None:
        recs = self._run([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        combined = " ".join(recs).lower()
        assert "calibrat" in combined

    def test_no_duplicate_recommendations(self) -> None:
        recs = self._run([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        assert len(recs) == len(set(recs))

    def test_low_gap_positive_feedback(self) -> None:
        recs = self._run([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])
        combined = " ".join(recs).lower()
        assert "proceed" in combined or "good proxy" in combined or "low" in combined


# ---------------------------------------------------------------------------
# CLI — gap estimate
# ---------------------------------------------------------------------------


class TestCliGapEstimate:
    def test_estimate_text_format(self) -> None:
        dims = {"velocity": {"sim": [0.2, 0.3, 0.3, 0.2], "real": [0.25, 0.25, 0.25, 0.25]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            result = _runner().invoke(
                cli,
                ["gap", "estimate", "--sim-data", sim_path, "--real-data", real_path],
            )
        assert result.exit_code == 0

    def test_estimate_json_format(self) -> None:
        dims = {"x": {"sim": [0.5, 0.5], "real": [0.5, 0.5]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--format", "json",
                ],
            )
        assert result.exit_code == 0
        # Output must contain valid JSON
        # Rich may wrap it, so check for key indicators
        assert "report_id" in result.output

    def test_estimate_markdown_format(self) -> None:
        dims = {"x": {"sim": [0.5, 0.5], "real": [0.5, 0.5]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--format", "markdown",
                ],
            )
        assert result.exit_code == 0
        assert "Gap Report" in result.output

    def test_estimate_with_metrics_flag(self) -> None:
        dims = {"x": {"sim": [0.4, 0.6], "real": [0.5, 0.5]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--metrics", "kl,wasserstein",
                ],
            )
        assert result.exit_code == 0

    def test_estimate_invalid_metric_exits_nonzero(self) -> None:
        dims = {"x": {"sim": [0.5, 0.5], "real": [0.5, 0.5]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--metrics", "badmetric",
                ],
            )
        assert result.exit_code != 0

    def test_estimate_missing_sim_file_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real_path = str(Path(tmp) / "real.json")
            Path(real_path).write_text('{"dimensions": {}}', encoding="utf-8")
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", "/nonexistent/sim.json",
                    "--real-data", real_path,
                ],
            )
        assert result.exit_code != 0

    def test_estimate_bad_json_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text("{not json", encoding="utf-8")
            good_path = Path(tmp) / "good.json"
            good_path.write_text('{"dimensions": {"x": {"sim": [1.0], "real": [1.0]}}}', encoding="utf-8")
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", str(bad_path),
                    "--real-data", str(good_path),
                ],
            )
        assert result.exit_code != 0

    def test_estimate_empty_dimensions_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim.json"
            real_path = Path(tmp) / "real.json"
            sim_path.write_text('{"dimensions": {}}', encoding="utf-8")
            real_path.write_text('{"dimensions": {}}', encoding="utf-8")
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", str(sim_path),
                    "--real-data", str(real_path),
                ],
            )
        assert result.exit_code != 0

    def test_estimate_no_common_dimensions_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim.json"
            real_path = Path(tmp) / "real.json"
            sim_path.write_text('{"dimensions": {"a": {"sim": [1.0], "real": [1.0]}}}', encoding="utf-8")
            real_path.write_text('{"dimensions": {"b": {"sim": [1.0], "real": [1.0]}}}', encoding="utf-8")
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", str(sim_path),
                    "--real-data", str(real_path),
                ],
            )
        assert result.exit_code != 0

    def test_estimate_all_single_metric_jsd(self) -> None:
        dims = {"x": {"sim": [0.5, 0.5], "real": [0.3, 0.7]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--metrics", "jsd",
                ],
            )
        assert result.exit_code == 0

    def test_estimate_missing_sim_key_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim.json"
            real_path = Path(tmp) / "real.json"
            # 'x' entry missing the nested 'sim' key
            sim_path.write_text('{"dimensions": {"x": {"no_sim_key": [1.0]}}}', encoding="utf-8")
            real_path.write_text('{"dimensions": {"x": {"real": [1.0]}}}', encoding="utf-8")
            result = _runner().invoke(
                cli,
                [
                    "gap", "estimate",
                    "--sim-data", str(sim_path),
                    "--real-data", str(real_path),
                ],
            )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI — gap compare
# ---------------------------------------------------------------------------


class TestCliGapCompare:
    def test_compare_creates_output_file(self) -> None:
        dims = {"torque": {"sim": [0.2, 0.3, 0.3, 0.2], "real": [0.25, 0.25, 0.25, 0.25]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            output_path = str(Path(tmp) / "report.json")
            result = _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--output", output_path,
                ],
            )
            assert result.exit_code == 0
            assert Path(output_path).exists()

    def test_compare_output_is_valid_json(self) -> None:
        dims = {"x": {"sim": [0.5, 0.5], "real": [0.5, 0.5]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            output_path = str(Path(tmp) / "report.json")
            _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--output", output_path,
                ],
            )
            content = Path(output_path).read_text(encoding="utf-8")
            parsed = json.loads(content)
            assert "report_id" in parsed

    def test_compare_output_contains_overall_score(self) -> None:
        dims = {"x": {"sim": [0.5, 0.5], "real": [0.5, 0.5]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            output_path = str(Path(tmp) / "report.json")
            _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--output", output_path,
                ],
            )
            parsed = json.loads(Path(output_path).read_text(encoding="utf-8"))
            assert "overall_score" in parsed

    def test_compare_shows_score_in_stdout(self) -> None:
        dims = {"x": {"sim": [0.5, 0.5], "real": [0.5, 0.5]}}
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            output_path = str(Path(tmp) / "report.json")
            result = _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--output", output_path,
                ],
            )
        assert "Overall gap score" in result.output or "score" in result.output.lower()

    def test_compare_missing_sim_file_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real_path = str(Path(tmp) / "real.json")
            Path(real_path).write_text('{"dimensions": {}}', encoding="utf-8")
            output_path = str(Path(tmp) / "out.json")
            result = _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", "/does/not/exist.json",
                    "--real-data", real_path,
                    "--output", output_path,
                ],
            )
        assert result.exit_code != 0

    def test_compare_bad_json_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text("{bad}", encoding="utf-8")
            good_path = Path(tmp) / "good.json"
            good_path.write_text('{"dimensions": {"x": {"sim": [1.0], "real": [1.0]}}}', encoding="utf-8")
            output_path = str(Path(tmp) / "out.json")
            result = _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", str(bad_path),
                    "--real-data", str(good_path),
                    "--output", output_path,
                ],
            )
        assert result.exit_code != 0

    def test_compare_empty_dimensions_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim.json"
            real_path = Path(tmp) / "real.json"
            sim_path.write_text('{"dimensions": {}}', encoding="utf-8")
            real_path.write_text('{"dimensions": {}}', encoding="utf-8")
            output_path = str(Path(tmp) / "out.json")
            result = _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", str(sim_path),
                    "--real-data", str(real_path),
                    "--output", output_path,
                ],
            )
        assert result.exit_code != 0

    def test_compare_multiple_dimensions(self) -> None:
        dims = {
            "dim_a": {"sim": [0.3, 0.4, 0.3], "real": [0.33, 0.34, 0.33]},
            "dim_b": {"sim": [0.5, 0.5], "real": [0.6, 0.4]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, real_path = _write_gap_data(tmp, dims, dims)
            output_path = str(Path(tmp) / "report.json")
            result = _runner().invoke(
                cli,
                [
                    "gap", "compare",
                    "--sim-data", sim_path,
                    "--real-data", real_path,
                    "--output", output_path,
                ],
            )
            assert result.exit_code == 0
            parsed = json.loads(Path(output_path).read_text(encoding="utf-8"))
            assert "dim_a" in parsed["dimensions"]
            assert "dim_b" in parsed["dimensions"]


# ---------------------------------------------------------------------------
# Gap package __init__ public API
# ---------------------------------------------------------------------------


class TestGapPackagePublicApi:
    def test_imports_from_gap_package(self) -> None:
        from agent_sim_bridge.gap import (  # noqa: F401
            DimensionGap,
            GapDimension,
            GapEstimator,
            GapMetric,
            GapReport,
            GapReporter,
            descriptive_stats,
            jensen_shannon_divergence,
            kl_divergence,
            maximum_mean_discrepancy,
            normalize_distribution,
            wasserstein_distance_1d,
        )

    def test_gap_group_in_cli_help(self) -> None:
        result = _runner().invoke(cli, ["--help"])
        assert "gap" in result.output.lower()

    def test_gap_estimate_in_help(self) -> None:
        result = _runner().invoke(cli, ["gap", "--help"])
        assert "estimate" in result.output.lower()

    def test_gap_compare_in_help(self) -> None:
        result = _runner().invoke(cli, ["gap", "--help"])
        assert "compare" in result.output.lower()
