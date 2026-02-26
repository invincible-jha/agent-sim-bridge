"""Unit tests for transfer modules.

Covers:
- transfer/bridge.py: CalibrationProfile (__post_init__ validation, n_dims),
  TransferBridge (sim_to_real, real_to_sim, length check, repr)
- transfer/calibration.py: _ols_slope_intercept (normal, empty, degenerate),
  Calibrator (calibrate, evaluate_calibration, last_profile, repr)
- transfer/domain_randomization.py: RandomizationConfig.sample (all
  distributions, clipping), DomainRandomizer (randomize, apply_randomization,
  seeded reproducibility, repr)
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_sim_bridge.transfer.bridge import CalibrationProfile, TransferBridge
from agent_sim_bridge.transfer.calibration import Calibrator, _ols_slope_intercept
from agent_sim_bridge.transfer.domain_randomization import (
    DistributionType,
    DomainRandomizer,
    RandomizationConfig,
)


# ---------------------------------------------------------------------------
# CalibrationProfile
# ---------------------------------------------------------------------------


class TestCalibrationProfile:
    def test_valid_profile_created(self) -> None:
        profile = CalibrationProfile(scale_factors=[1.0, 2.0], offsets=[0.0, -1.0])
        assert profile.n_dims == 2

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="must equal"):
            CalibrationProfile(scale_factors=[1.0, 2.0], offsets=[0.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            CalibrationProfile(scale_factors=[], offsets=[])

    def test_zero_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            CalibrationProfile(scale_factors=[0.0, 1.0], offsets=[0.0, 0.0])

    def test_n_dims_property(self) -> None:
        profile = CalibrationProfile(scale_factors=[1.0, 2.0, 3.0], offsets=[0.0, 0.0, 0.0])
        assert profile.n_dims == 3

    def test_noise_model_stored(self) -> None:
        profile = CalibrationProfile(
            scale_factors=[1.0], offsets=[0.0], noise_model="gaussian"
        )
        assert profile.noise_model == "gaussian"


# ---------------------------------------------------------------------------
# TransferBridge
# ---------------------------------------------------------------------------


class TestTransferBridge:
    def setup_method(self) -> None:
        self.profile = CalibrationProfile(
            scale_factors=[2.0, 0.5], offsets=[1.0, -0.5]
        )
        self.bridge = TransferBridge(self.profile)

    def test_profile_property(self) -> None:
        assert self.bridge.profile is self.profile

    def test_sim_to_real(self) -> None:
        # dim0: 1.0 * 2.0 + 1.0 = 3.0; dim1: 2.0 * 0.5 + (-0.5) = 0.5
        result = self.bridge.sim_to_real([1.0, 2.0])
        assert result == pytest.approx([3.0, 0.5])

    def test_real_to_sim(self) -> None:
        # Inverse of sim_to_real([1.0, 2.0]) = [3.0, 0.5]
        result = self.bridge.real_to_sim([3.0, 0.5])
        assert result == pytest.approx([1.0, 2.0])

    def test_sim_to_real_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Expected 2 values"):
            self.bridge.sim_to_real([1.0])

    def test_real_to_sim_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Expected 2 values"):
            self.bridge.real_to_sim([1.0, 2.0, 3.0])

    def test_roundtrip(self) -> None:
        original = [0.3, -1.7]
        real = self.bridge.sim_to_real(original)
        back = self.bridge.real_to_sim(real)
        assert back == pytest.approx(original, abs=1e-6)

    def test_repr(self) -> None:
        result = repr(self.bridge)
        assert "TransferBridge" in result
        assert "2" in result


# ---------------------------------------------------------------------------
# _ols_slope_intercept
# ---------------------------------------------------------------------------


class TestOlsSlopeIntercept:
    def test_perfect_linear_fit(self) -> None:
        # y = 2x + 1
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [3.0, 5.0, 7.0, 9.0]
        slope, intercept = _ols_slope_intercept(xs, ys)
        assert slope == pytest.approx(2.0, abs=1e-6)
        assert intercept == pytest.approx(1.0, abs=1e-6)

    def test_empty_returns_identity(self) -> None:
        slope, intercept = _ols_slope_intercept([], [])
        assert slope == 1.0
        assert intercept == 0.0

    def test_degenerate_all_same_x(self) -> None:
        # All x values identical — denominator is zero.
        xs = [2.0, 2.0, 2.0]
        ys = [5.0, 7.0, 9.0]
        slope, intercept = _ols_slope_intercept(xs, ys)
        # slope forced to 1.0, intercept = mean(y) - mean(x)
        assert slope == 1.0
        assert intercept == pytest.approx(7.0 - 2.0, abs=1e-6)

    def test_single_point(self) -> None:
        # One sample — mathematically singular but should not crash.
        slope, intercept = _ols_slope_intercept([3.0], [6.0])
        # n=1: denominator = 1*9 - 9 = 0 → degenerate path
        assert slope == 1.0
        assert intercept == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class TestCalibrator:
    def _paired_data(
        self, n: int = 10, scale: float = 2.0, offset: float = 0.5
    ) -> tuple[list[list[float]], list[list[float]]]:
        sim = [[float(i), float(i) * 0.5] for i in range(n)]
        real = [[s[0] * scale + offset, s[1] * scale + offset] for s in sim]
        return sim, real

    def test_calibrate_returns_profile(self) -> None:
        sim, real = self._paired_data()
        cal = Calibrator()
        profile = cal.calibrate(sim, real)
        assert isinstance(profile, CalibrationProfile)
        assert profile.n_dims == 2

    def test_calibrate_recovers_transform(self) -> None:
        # y = 2x + 0.5 exactly.
        sim = [[float(i)] for i in range(10)]
        real = [[s[0] * 2.0 + 0.5] for s in sim]
        cal = Calibrator()
        profile = cal.calibrate(sim, real)
        assert profile.scale_factors[0] == pytest.approx(2.0, abs=1e-4)
        assert profile.offsets[0] == pytest.approx(0.5, abs=1e-4)

    def test_calibrate_stores_last_profile(self) -> None:
        sim, real = self._paired_data()
        cal = Calibrator()
        profile = cal.calibrate(sim, real)
        assert cal.last_profile is profile

    def test_calibrate_mismatched_lengths_raises(self) -> None:
        cal = Calibrator()
        with pytest.raises(ValueError, match="samples but real_obs has"):
            cal.calibrate([[1.0]], [[1.0], [2.0]])

    def test_calibrate_empty_raises(self) -> None:
        cal = Calibrator()
        with pytest.raises(ValueError, match="zero samples"):
            cal.calibrate([], [])

    def test_calibrate_dim_mismatch_raises(self) -> None:
        cal = Calibrator()
        with pytest.raises(ValueError, match="dimensions"):
            cal.calibrate([[1.0, 2.0], [1.0]], [[1.0, 2.0], [1.0, 2.0]])

    def test_calibrate_real_dim_mismatch_raises(self) -> None:
        cal = Calibrator()
        with pytest.raises(ValueError, match="dimensions"):
            cal.calibrate([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0]])

    def test_evaluate_calibration_with_last_profile(self) -> None:
        sim, real = self._paired_data()
        cal = Calibrator()
        cal.calibrate(sim, real)
        error = cal.evaluate_calibration(sim, real)
        assert error >= 0.0
        # For a well-fitted profile, MAE should be very small.
        assert error < 0.01

    def test_evaluate_calibration_with_explicit_profile(self) -> None:
        profile = CalibrationProfile(scale_factors=[1.0], offsets=[0.0])
        cal = Calibrator()
        sim = [[1.0], [2.0], [3.0]]
        real = [[1.0], [2.0], [3.0]]
        error = cal.evaluate_calibration(sim, real, profile=profile)
        assert error == pytest.approx(0.0)

    def test_evaluate_calibration_no_profile_raises(self) -> None:
        cal = Calibrator()
        with pytest.raises(RuntimeError, match="No calibration profile"):
            cal.evaluate_calibration([[1.0]], [[1.0]])

    def test_evaluate_calibration_length_mismatch_raises(self) -> None:
        cal = Calibrator()
        cal.calibrate([[1.0]], [[1.0]])
        with pytest.raises(ValueError):
            cal.evaluate_calibration([[1.0]], [[1.0], [2.0]])

    def test_evaluate_calibration_empty_returns_zero(self) -> None:
        sim, real = self._paired_data()
        cal = Calibrator()
        cal.calibrate(sim, real)
        error = cal.evaluate_calibration([], [])
        assert error == 0.0

    def test_noise_model_embedded_in_profile(self) -> None:
        cal = Calibrator(noise_model="gaussian")
        sim, real = self._paired_data()
        profile = cal.calibrate(sim, real)
        assert profile.noise_model == "gaussian"

    def test_repr_unfitted(self) -> None:
        cal = Calibrator()
        result = repr(cal)
        assert "fitted=False" in result

    def test_repr_fitted(self) -> None:
        sim, real = self._paired_data()
        cal = Calibrator()
        cal.calibrate(sim, real)
        result = repr(cal)
        assert "fitted=True" in result

    def test_zero_slope_replaced_with_one(self) -> None:
        # When OLS gives slope=0 (flat y), calibrator replaces with 1.0.
        sim = [[1.0], [2.0], [3.0]]
        real = [[5.0], [5.0], [5.0]]  # constant real → slope=0
        cal = Calibrator()
        profile = cal.calibrate(sim, real)
        assert profile.scale_factors[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RandomizationConfig.sample
# ---------------------------------------------------------------------------


class TestRandomizationConfigSample:
    def test_constant_returns_low(self) -> None:
        config = RandomizationConfig(
            parameter_name="p",
            distribution=DistributionType.CONSTANT,
            low=3.14,
        )
        for _ in range(5):
            assert config.sample() == pytest.approx(3.14)

    def test_uniform_in_range(self) -> None:
        config = RandomizationConfig(
            parameter_name="p",
            distribution=DistributionType.UNIFORM,
            low=2.0,
            high=5.0,
        )
        for _ in range(20):
            v = config.sample()
            assert 2.0 <= v <= 5.0

    def test_gaussian_has_variance(self) -> None:
        config = RandomizationConfig(
            parameter_name="p",
            distribution=DistributionType.GAUSSIAN,
            low=0.0,
            high=1.0,
        )
        samples = [config.sample() for _ in range(100)]
        assert max(samples) != min(samples)

    def test_log_uniform_positive(self) -> None:
        config = RandomizationConfig(
            parameter_name="p",
            distribution=DistributionType.LOG_UNIFORM,
            low=0.1,
            high=10.0,
        )
        for _ in range(20):
            v = config.sample()
            assert v > 0.0
            assert 0.1 <= v <= 10.0

    def test_log_uniform_invalid_low_raises(self) -> None:
        config = RandomizationConfig(
            parameter_name="p",
            distribution=DistributionType.LOG_UNIFORM,
            low=0.0,
            high=1.0,
        )
        with pytest.raises(ValueError, match="LOG_UNIFORM requires low > 0"):
            config.sample()

    def test_clip_low_enforced(self) -> None:
        config = RandomizationConfig(
            parameter_name="p",
            distribution=DistributionType.UNIFORM,
            low=0.0,
            high=1.0,
            clip_low=0.5,
        )
        for _ in range(20):
            assert config.sample() >= 0.5

    def test_clip_high_enforced(self) -> None:
        config = RandomizationConfig(
            parameter_name="p",
            distribution=DistributionType.UNIFORM,
            low=0.0,
            high=10.0,
            clip_high=2.0,
        )
        for _ in range(20):
            assert config.sample() <= 2.0


# ---------------------------------------------------------------------------
# DomainRandomizer
# ---------------------------------------------------------------------------


class TestDomainRandomizer:
    def test_randomize_empty_configs(self) -> None:
        randomizer = DomainRandomizer(seed=1)
        result = randomizer.randomize([])
        assert result == {}

    def test_randomize_returns_all_params(self) -> None:
        configs = [
            RandomizationConfig("gravity", DistributionType.UNIFORM, low=8.0, high=12.0),
            RandomizationConfig("friction", DistributionType.CONSTANT, low=0.5),
        ]
        randomizer = DomainRandomizer(seed=42)
        result = randomizer.randomize(configs)
        assert "gravity" in result
        assert "friction" in result
        assert result["friction"] == pytest.approx(0.5)

    def test_seeded_randomize_is_reproducible(self) -> None:
        configs = [
            RandomizationConfig("mass", DistributionType.UNIFORM, low=1.0, high=5.0),
        ]
        r1 = DomainRandomizer(seed=7)
        r2 = DomainRandomizer(seed=7)
        assert r1.randomize(configs)["mass"] == pytest.approx(
            r2.randomize(configs)["mass"]
        )

    def test_unseed_randomizer(self) -> None:
        configs = [
            RandomizationConfig("gravity", DistributionType.UNIFORM, low=9.0, high=9.8),
        ]
        randomizer = DomainRandomizer(seed=None)
        result = randomizer.randomize(configs)
        assert "gravity" in result

    def test_apply_randomization_sets_attribute(self) -> None:
        env = SimpleNamespace(gravity=9.8, friction=0.5)
        configs = [
            RandomizationConfig("gravity", DistributionType.CONSTANT, low=11.0),
        ]
        randomizer = DomainRandomizer(seed=1)
        applied = randomizer.apply_randomization(env, configs)
        assert env.gravity == pytest.approx(11.0)
        assert applied["gravity"] == pytest.approx(11.0)

    def test_apply_randomization_nested_attribute(self) -> None:
        # parameter_name = "physics.gravity" traverses nested attribute.
        physics = SimpleNamespace(gravity=9.8)
        env = SimpleNamespace(physics=physics)
        configs = [
            RandomizationConfig("physics.gravity", DistributionType.CONSTANT, low=5.0),
        ]
        randomizer = DomainRandomizer(seed=1)
        randomizer.apply_randomization(env, configs)
        assert env.physics.gravity == pytest.approx(5.0)

    def test_apply_randomization_missing_attribute_raises(self) -> None:
        # A dot-path like "missing_obj.attr" causes getattr(env, "missing_obj")
        # to raise AttributeError when "missing_obj" does not exist on env.
        env = SimpleNamespace()
        configs = [
            RandomizationConfig("missing_obj.attr", DistributionType.CONSTANT, low=1.0),
        ]
        randomizer = DomainRandomizer(seed=1)
        with pytest.raises(AttributeError):
            randomizer.apply_randomization(env, configs)

    def test_repr(self) -> None:
        randomizer = DomainRandomizer(seed=99)
        result = repr(randomizer)
        assert "99" in result
        assert "DomainRandomizer" in result

    def test_stdlib_rng_state_restored_after_randomize(self) -> None:
        """randomize() must not alter the global stdlib random state."""
        random.seed(0)
        state_before = random.getstate()
        configs = [
            RandomizationConfig("x", DistributionType.UNIFORM, low=0.0, high=1.0),
        ]
        DomainRandomizer(seed=42).randomize(configs)
        state_after = random.getstate()
        assert state_before == state_after
