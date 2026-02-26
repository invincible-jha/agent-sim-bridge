"""Unit tests for agent-sim-bridge metrics and sensor modules.

Covers:
- GapReport and SimRealGap (metrics/gap.py)
- MetricRecord, _statistics, and PerformanceTracker (metrics/performance.py)
- SafetyMonitor and MonitoredStep (safety/monitor.py)
- SensorType, SensorReading, and Sensor ABC (sensors/base.py)
- FusionStrategy and SensorFusion (sensors/fusion.py)
- GaussianNoise, UniformNoise, CompositeNoise (sensors/noise.py)
"""
from __future__ import annotations

import math

import pytest

from agent_sim_bridge.metrics.gap import GapReport, SimRealGap
from agent_sim_bridge.metrics.performance import (
    MetricRecord,
    PerformanceTracker,
    _statistics,
)
from agent_sim_bridge.safety.constraints import (
    ConstraintType,
    SafetyConstraint,
    ViolationSeverity,
)
from agent_sim_bridge.safety.monitor import MonitoredStep, SafetyMonitor
from agent_sim_bridge.sensors.base import Sensor, SensorReading, SensorType
from agent_sim_bridge.sensors.fusion import FusionStrategy, SensorFusion
from agent_sim_bridge.sensors.noise import CompositeNoise, GaussianNoise, UniformNoise


# ---------------------------------------------------------------------------
# Helpers / concrete subclasses
# ---------------------------------------------------------------------------


class ConstantSensor(Sensor):
    """A sensor that always returns fixed values."""

    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        values: list[float],
        confidence: float = 1.0,
    ) -> None:
        super().__init__(sensor_id, sensor_type)
        self._values = values
        self._confidence = confidence

    def read(self) -> SensorReading:
        return SensorReading(
            sensor_id=self._sensor_id,
            sensor_type=self._sensor_type,
            values=list(self._values),
            confidence=self._confidence,
        )


def _make_reading(
    sensor_id: str = "s1",
    sensor_type: SensorType = SensorType.POSITION,
    values: list[float] | None = None,
    confidence: float = 1.0,
) -> SensorReading:
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        values=values or [1.0, 2.0, 3.0],
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# GapReport
# ---------------------------------------------------------------------------


class TestGapReport:
    def test_default_values(self) -> None:
        report = GapReport()
        assert report.observation_mae == 0.0
        assert report.observation_rmse == 0.0
        assert report.reward_mae == 0.0
        assert report.trajectory_length_ratio == 1.0
        assert report.overall_gap_score == 0.0

    def test_summary_returns_dict_with_all_keys(self) -> None:
        report = GapReport()
        summary = report.summary()
        for key in (
            "observation_mae",
            "observation_rmse",
            "reward_mae",
            "reward_bias",
            "trajectory_length_ratio",
            "overall_gap_score",
            "n_sim_steps",
            "n_real_steps",
        ):
            assert key in summary

    def test_compute_overall_score_no_error(self) -> None:
        report = GapReport(observation_mae=0.1, reward_mae=0.2, trajectory_length_ratio=0.9)
        score = report.compute_overall_score()
        assert score >= 0.0
        assert report.overall_gap_score == pytest.approx(score)

    def test_compute_overall_score_zero_weights_returns_zero(self) -> None:
        report = GapReport(observation_mae=0.5)
        score = report.compute_overall_score(
            obs_weight=0.0, reward_weight=0.0, length_weight=0.0
        )
        assert score == pytest.approx(0.0)

    def test_compute_overall_score_perfect_scenario(self) -> None:
        # Zero errors, length ratio=1.0 (ideal) → score should be 0
        report = GapReport(observation_mae=0.0, reward_mae=0.0, trajectory_length_ratio=1.0)
        score = report.compute_overall_score()
        assert score == pytest.approx(0.0)

    def test_metadata_preserved(self) -> None:
        report = GapReport(metadata={"policy": "ppo-v2"})
        assert report.metadata["policy"] == "ppo-v2"


# ---------------------------------------------------------------------------
# SimRealGap
# ---------------------------------------------------------------------------


class TestSimRealGap:
    def test_repr(self) -> None:
        gap = SimRealGap(metadata={"x": 1})
        assert "SimRealGap" in repr(gap)

    def test_no_metadata(self) -> None:
        gap = SimRealGap()
        assert gap._metadata == {}

    def test_empty_observations_returns_zero_report(self) -> None:
        gap = SimRealGap()
        report = gap.measure_gap([], [])
        assert report.observation_mae == 0.0
        assert report.n_sim_steps == 0

    def test_matched_observations_computes_mae(self) -> None:
        gap = SimRealGap()
        sim_obs = [[1.0, 2.0], [3.0, 4.0]]
        real_obs = [[1.1, 2.1], [3.1, 4.1]]
        report = gap.measure_gap(sim_obs, real_obs)
        assert report.observation_mae == pytest.approx(0.1, abs=1e-6)

    def test_identical_observations_zero_mae(self) -> None:
        gap = SimRealGap()
        obs = [[1.0, 2.0, 3.0]]
        report = gap.measure_gap(obs, obs)
        assert report.observation_mae == pytest.approx(0.0)

    def test_computes_rmse(self) -> None:
        gap = SimRealGap()
        sim_obs = [[2.0]]
        real_obs = [[0.0]]  # Error = 2.0
        report = gap.measure_gap(sim_obs, real_obs)
        assert report.observation_rmse == pytest.approx(2.0)

    def test_reward_statistics_computed(self) -> None:
        gap = SimRealGap()
        obs = [[1.0]]
        report = gap.measure_gap(obs, obs, sim_rewards=[1.0], real_rewards=[0.5])
        assert report.reward_mae == pytest.approx(0.5)
        assert report.reward_bias == pytest.approx(0.5)

    def test_no_rewards_reward_mae_zero(self) -> None:
        gap = SimRealGap()
        obs = [[1.0]]
        report = gap.measure_gap(obs, obs)
        assert report.reward_mae == 0.0

    def test_trajectory_length_ratio_computed(self) -> None:
        gap = SimRealGap()
        sim_obs = [[1.0]] * 4
        real_obs = [[1.0]] * 2
        report = gap.measure_gap(sim_obs, real_obs)
        assert report.trajectory_length_ratio == pytest.approx(0.5)

    def test_mismatched_lengths_uses_min(self) -> None:
        gap = SimRealGap()
        sim_obs = [[1.0]] * 5
        real_obs = [[1.0]] * 3
        report = gap.measure_gap(sim_obs, real_obs)
        assert report.n_sim_steps == 5
        assert report.n_real_steps == 3

    def test_metadata_in_report(self) -> None:
        gap = SimRealGap(metadata={"env": "MuJoCo"})
        report = gap.measure_gap([[1.0]], [[1.0]])
        assert report.metadata["env"] == "MuJoCo"

    def test_overall_score_computed(self) -> None:
        gap = SimRealGap()
        report = gap.measure_gap([[1.0]], [[2.0]])
        assert report.overall_gap_score > 0.0


# ---------------------------------------------------------------------------
# _statistics helper
# ---------------------------------------------------------------------------


class TestStatisticsHelper:
    def test_empty_returns_nan_stats(self) -> None:
        stats = _statistics([])
        assert stats["count"] == 0.0
        assert math.isnan(stats["min"])
        assert math.isnan(stats["max"])
        assert math.isnan(stats["mean"])
        assert math.isnan(stats["std"])

    def test_single_value(self) -> None:
        stats = _statistics([5.0])
        assert stats["count"] == 1.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["mean"] == pytest.approx(5.0)
        assert stats["std"] == pytest.approx(0.0)

    def test_multiple_values(self) -> None:
        stats = _statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["count"] == 5.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == pytest.approx(3.0)

    def test_std_non_zero_for_spread(self) -> None:
        stats = _statistics([0.0, 10.0])
        assert stats["std"] > 0.0


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------


class TestPerformanceTrackerRecord:
    def setup_method(self) -> None:
        self.tracker = PerformanceTracker()

    def test_record_single_metric(self) -> None:
        self.tracker.record("reward", 42.0)
        assert self.tracker.get_values("reward") == [42.0]

    def test_record_multiple_values(self) -> None:
        for i in range(5):
            self.tracker.record("step_count", float(i))
        assert len(self.tracker.get_values("step_count")) == 5

    def test_record_with_step(self) -> None:
        self.tracker.record("loss", 0.5, step=10)
        records = self.tracker.get_records("loss")
        assert records[0].step == 10

    def test_record_with_tags(self) -> None:
        self.tracker.record("loss", 0.5, tags=["train"])
        records = self.tracker.get_records("loss")
        assert "train" in records[0].tags

    def test_record_many(self) -> None:
        self.tracker.record_many({"loss": 0.5, "accuracy": 0.9}, step=1)
        assert self.tracker.get_values("loss") == [0.5]
        assert self.tracker.get_values("accuracy") == [0.9]

    def test_max_records_evicts_oldest(self) -> None:
        tracker = PerformanceTracker(max_records_per_metric=3)
        for i in range(5):
            tracker.record("x", float(i))
        values = tracker.get_values("x")
        assert len(values) == 3
        assert values == [2.0, 3.0, 4.0]

    def test_get_values_unknown_returns_empty(self) -> None:
        assert self.tracker.get_values("nonexistent") == []

    def test_get_records_unknown_returns_empty(self) -> None:
        assert self.tracker.get_records("nonexistent") == []


class TestPerformanceTrackerSummary:
    def setup_method(self) -> None:
        self.tracker = PerformanceTracker()
        self.tracker.record("reward", 10.0)
        self.tracker.record("reward", 20.0)
        self.tracker.record("loss", 0.5)

    def test_summary_returns_dict(self) -> None:
        summary = self.tracker.summary()
        assert isinstance(summary, dict)

    def test_summary_includes_all_metrics(self) -> None:
        summary = self.tracker.summary()
        assert "reward" in summary
        assert "loss" in summary

    def test_summary_statistics_correct(self) -> None:
        summary = self.tracker.summary()
        assert summary["reward"]["mean"] == pytest.approx(15.0)
        assert summary["reward"]["count"] == 2.0

    def test_metric_summary_single(self) -> None:
        stats = self.tracker.metric_summary("reward")
        assert stats["min"] == pytest.approx(10.0)
        assert stats["max"] == pytest.approx(20.0)

    def test_metric_summary_unknown_returns_nan(self) -> None:
        stats = self.tracker.metric_summary("unknown_metric")
        assert math.isnan(stats["mean"])

    def test_metric_names_sorted(self) -> None:
        names = self.tracker.metric_names()
        assert names == sorted(names)

    def test_len_returns_metric_count(self) -> None:
        assert len(self.tracker) == 2

    def test_repr_includes_metrics(self) -> None:
        assert "PerformanceTracker" in repr(self.tracker)


class TestPerformanceTrackerReset:
    def test_reset_all(self) -> None:
        tracker = PerformanceTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        tracker.reset()
        assert len(tracker) == 0

    def test_reset_single_metric(self) -> None:
        tracker = PerformanceTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        tracker.reset("a")
        assert "a" not in tracker.metric_names()
        assert "b" in tracker.metric_names()

    def test_reset_unknown_metric_no_error(self) -> None:
        tracker = PerformanceTracker()
        tracker.reset("nonexistent")  # Should not raise


# ---------------------------------------------------------------------------
# SafetyMonitor
# ---------------------------------------------------------------------------


def _make_range_constraint(
    name: str = "joint",
    dimension: int = 0,
    min_value: float = -1.0,
    max_value: float = 1.0,
    severity: ViolationSeverity = ViolationSeverity.ERROR,
) -> SafetyConstraint:
    return SafetyConstraint(
        name=name,
        constraint_type=ConstraintType.RANGE,
        dimension=dimension,
        min_value=min_value,
        max_value=max_value,
        severity=severity,
    )


class TestSafetyMonitorLifecycle:
    def setup_method(self) -> None:
        self.constraint = _make_range_constraint()
        self.monitor = SafetyMonitor([self.constraint])

    def test_not_monitoring_by_default(self) -> None:
        assert self.monitor.is_monitoring is False

    def test_start_monitoring(self) -> None:
        self.monitor.start_monitoring()
        assert self.monitor.is_monitoring is True

    def test_stop_monitoring(self) -> None:
        self.monitor.start_monitoring()
        self.monitor.stop_monitoring()
        assert self.monitor.is_monitoring is False

    def test_step_count_zero_before_start(self) -> None:
        assert self.monitor.step_count == 0

    def test_step_count_increments(self) -> None:
        self.monitor.start_monitoring()
        self.monitor.check_step([0.5])
        assert self.monitor.step_count == 1

    def test_check_step_without_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not active"):
            self.monitor.check_step([0.5])

    def test_start_resets_state(self) -> None:
        self.monitor.start_monitoring()
        self.monitor.check_step([2.0])  # violation
        self.monitor.start_monitoring()  # Reset
        assert self.monitor.step_count == 0
        assert len(self.monitor.get_violations()) == 0


class TestSafetyMonitorCheckStep:
    def setup_method(self) -> None:
        self.constraint = _make_range_constraint(min_value=-1.0, max_value=1.0)
        self.monitor = SafetyMonitor([self.constraint])
        self.monitor.start_monitoring()

    def test_safe_action_returns_empty_violations(self) -> None:
        violations = self.monitor.check_step([0.5])
        assert violations == []

    def test_unsafe_action_returns_violations(self) -> None:
        violations = self.monitor.check_step([5.0])
        assert len(violations) == 1

    def test_violations_accumulated(self) -> None:
        self.monitor.check_step([0.5])
        self.monitor.check_step([5.0])
        assert len(self.monitor.get_violations()) == 1

    def test_get_step_records_populated(self) -> None:
        self.monitor.check_step([0.5])
        records = self.monitor.get_step_records()
        assert len(records) == 1
        assert isinstance(records[0], MonitoredStep)

    def test_step_record_contains_action(self) -> None:
        self.monitor.check_step([0.7])
        record = self.monitor.get_step_records()[0]
        assert record.action == [0.7]

    def test_step_record_step_index(self) -> None:
        self.monitor.check_step([0.5])
        self.monitor.check_step([0.5])
        records = self.monitor.get_step_records()
        assert records[0].step_index == 0
        assert records[1].step_index == 1


class TestSafetyMonitorEmergencyStop:
    def test_emergency_stop_not_triggered_by_default(self) -> None:
        constraint = _make_range_constraint(severity=ViolationSeverity.ERROR)
        monitor = SafetyMonitor([constraint], auto_stop_on_critical=True)
        monitor.start_monitoring()
        monitor.check_step([5.0])  # Violation but not CRITICAL
        assert monitor.emergency_stopped is False

    def test_critical_violation_triggers_emergency_stop(self) -> None:
        constraint = _make_range_constraint(severity=ViolationSeverity.CRITICAL)
        monitor = SafetyMonitor([constraint], auto_stop_on_critical=True)
        monitor.start_monitoring()
        monitor.check_step([5.0])  # CRITICAL violation
        assert monitor.emergency_stopped is True

    def test_manual_emergency_stop(self) -> None:
        monitor = SafetyMonitor([], auto_stop_on_critical=False)
        monitor.start_monitoring()
        monitor.emergency_stop()
        assert monitor.emergency_stopped is True

    def test_emergency_stop_idempotent(self) -> None:
        monitor = SafetyMonitor([])
        monitor.start_monitoring()
        monitor.emergency_stop()
        monitor.emergency_stop()  # Second call should not raise
        assert monitor.emergency_stopped is True

    def test_auto_stop_disabled(self) -> None:
        constraint = _make_range_constraint(severity=ViolationSeverity.CRITICAL)
        monitor = SafetyMonitor([constraint], auto_stop_on_critical=False)
        monitor.start_monitoring()
        monitor.check_step([5.0])  # CRITICAL but auto_stop disabled
        assert monitor.emergency_stopped is False

    def test_check_step_after_emergency_still_works(self) -> None:
        constraint = _make_range_constraint(severity=ViolationSeverity.CRITICAL)
        monitor = SafetyMonitor([constraint], auto_stop_on_critical=True)
        monitor.start_monitoring()
        monitor.check_step([5.0])  # Triggers emergency stop
        # Subsequent check should still execute (with a warning)
        violations = monitor.check_step([0.0])
        assert isinstance(violations, list)


class TestSafetyMonitorSummary:
    def test_summary_keys(self) -> None:
        monitor = SafetyMonitor([])
        monitor.start_monitoring()
        summary = monitor.summary()
        assert "total_steps" in summary
        assert "total_violations" in summary
        assert "emergency_stopped" in summary
        assert "severity_counts" in summary

    def test_summary_counts_severity(self) -> None:
        constraint = _make_range_constraint(severity=ViolationSeverity.WARNING)
        monitor = SafetyMonitor([constraint])
        monitor.start_monitoring()
        monitor.check_step([5.0])
        summary = monitor.summary()
        assert summary["severity_counts"]["warning"] == 1

    def test_repr_includes_class_name(self) -> None:
        monitor = SafetyMonitor([])
        assert "SafetyMonitor" in repr(monitor)


# ---------------------------------------------------------------------------
# SensorReading
# ---------------------------------------------------------------------------


class TestSensorReading:
    def test_valid_reading(self) -> None:
        reading = SensorReading(
            sensor_id="s1",
            sensor_type=SensorType.POSITION,
            values=[1.0, 2.0, 3.0],
        )
        assert reading.sensor_id == "s1"
        assert len(reading.values) == 3

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SensorReading(
                sensor_id="s1",
                sensor_type=SensorType.POSITION,
                values=[1.0],
                confidence=1.1,
            )

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SensorReading(
                sensor_id="s1",
                sensor_type=SensorType.POSITION,
                values=[1.0],
                confidence=-0.1,
            )

    def test_empty_values_raises(self) -> None:
        with pytest.raises(ValueError, match="values"):
            SensorReading(
                sensor_id="s1",
                sensor_type=SensorType.POSITION,
                values=[],
            )

    def test_scalar_property_single_value(self) -> None:
        reading = SensorReading(
            sensor_id="s1",
            sensor_type=SensorType.TEMPERATURE,
            values=[42.0],
        )
        assert reading.scalar == pytest.approx(42.0)

    def test_scalar_property_multi_value_raises(self) -> None:
        reading = SensorReading(
            sensor_id="s1",
            sensor_type=SensorType.POSITION,
            values=[1.0, 2.0],
        )
        with pytest.raises(ValueError, match="multi-value"):
            _ = reading.scalar

    def test_default_confidence_is_one(self) -> None:
        reading = SensorReading(
            sensor_id="s1",
            sensor_type=SensorType.IMU,
            values=[0.1],
        )
        assert reading.confidence == 1.0


# ---------------------------------------------------------------------------
# Sensor ABC
# ---------------------------------------------------------------------------


class TestSensor:
    def test_concrete_sensor_readable(self) -> None:
        sensor = ConstantSensor("s1", SensorType.POSITION, [1.0, 2.0])
        reading = sensor.read()
        assert reading.sensor_id == "s1"
        assert reading.values == [1.0, 2.0]

    def test_sensor_id_property(self) -> None:
        sensor = ConstantSensor("my-sensor", SensorType.VELOCITY, [3.0])
        assert sensor.sensor_id == "my-sensor"

    def test_sensor_type_property(self) -> None:
        sensor = ConstantSensor("s1", SensorType.FORCE, [9.8])
        assert sensor.sensor_type == SensorType.FORCE

    def test_calibrate_default_noop(self) -> None:
        sensor = ConstantSensor("s1", SensorType.IMU, [0.0])
        sensor.calibrate()  # Should not raise

    def test_repr_includes_class_name(self) -> None:
        sensor = ConstantSensor("s1", SensorType.LIDAR, [1.0])
        assert "ConstantSensor" in repr(sensor)


# ---------------------------------------------------------------------------
# SensorFusion
# ---------------------------------------------------------------------------


class TestSensorFusionWeightedAverage:
    def setup_method(self) -> None:
        self.fusion = SensorFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)

    def test_empty_readings_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.fusion.fuse([])

    def test_single_reading_returns_fused(self) -> None:
        reading = _make_reading(values=[1.0, 2.0])
        result = self.fusion.fuse([reading])
        assert result.sensor_id == "fused"
        assert result.values == pytest.approx([1.0, 2.0])

    def test_two_equal_confidence_readings_averaged(self) -> None:
        r1 = _make_reading(sensor_id="s1", values=[0.0])
        r2 = _make_reading(sensor_id="s2", values=[2.0])
        result = self.fusion.fuse([r1, r2])
        assert result.values == pytest.approx([1.0])

    def test_high_confidence_reading_dominates(self) -> None:
        r1 = _make_reading(sensor_id="s1", values=[0.0], confidence=0.1)
        r2 = _make_reading(sensor_id="s2", values=[10.0], confidence=0.9)
        result = self.fusion.fuse([r1, r2])
        assert result.values[0] > 5.0  # Biased toward high-confidence reading

    def test_all_zero_confidence_falls_back_to_equal_weighting(self) -> None:
        r1 = _make_reading(sensor_id="s1", values=[0.0], confidence=0.0)
        r2 = _make_reading(sensor_id="s2", values=[4.0], confidence=0.0)
        result = self.fusion.fuse([r1, r2])
        assert result.values == pytest.approx([2.0])

    def test_inconsistent_dimensions_raises(self) -> None:
        r1 = _make_reading(values=[1.0, 2.0])
        r2 = _make_reading(values=[3.0])
        with pytest.raises(ValueError):
            self.fusion.fuse([r1, r2])

    def test_output_sensor_type_inferred_from_first(self) -> None:
        reading = _make_reading(sensor_type=SensorType.VELOCITY, values=[5.0])
        result = self.fusion.fuse([reading])
        assert result.sensor_type == SensorType.VELOCITY

    def test_output_sensor_type_overridden(self) -> None:
        fusion = SensorFusion(
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
            output_sensor_type=SensorType.FORCE,
        )
        reading = _make_reading(sensor_type=SensorType.POSITION, values=[1.0])
        result = fusion.fuse([reading])
        assert result.sensor_type == SensorType.FORCE


class TestSensorFusionSimpleAverage:
    def setup_method(self) -> None:
        self.fusion = SensorFusion(strategy=FusionStrategy.SIMPLE_AVERAGE)

    def test_simple_average_two_readings(self) -> None:
        r1 = _make_reading(sensor_id="s1", values=[0.0, 0.0])
        r2 = _make_reading(sensor_id="s2", values=[2.0, 4.0])
        result = self.fusion.fuse([r1, r2])
        assert result.values == pytest.approx([1.0, 2.0])

    def test_strategy_property(self) -> None:
        assert self.fusion.strategy == FusionStrategy.SIMPLE_AVERAGE


class TestSensorFusionHighestLowestConfidence:
    def test_highest_confidence_selects_correct_reading(self) -> None:
        fusion = SensorFusion(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        r1 = _make_reading(sensor_id="low", values=[1.0], confidence=0.2)
        r2 = _make_reading(sensor_id="high", values=[9.0], confidence=0.9)
        result = fusion.fuse([r1, r2])
        assert result.values == pytest.approx([9.0])
        assert result.metadata["source_sensor_id"] == "high"

    def test_lowest_confidence_selects_correct_reading(self) -> None:
        fusion = SensorFusion(strategy=FusionStrategy.LOWEST_CONFIDENCE)
        r1 = _make_reading(sensor_id="low", values=[1.0], confidence=0.2)
        r2 = _make_reading(sensor_id="high", values=[9.0], confidence=0.9)
        result = fusion.fuse([r1, r2])
        assert result.values == pytest.approx([1.0])

    def test_repr_includes_strategy(self) -> None:
        fusion = SensorFusion(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        assert "highest_confidence" in repr(fusion)


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------


class TestGaussianNoise:
    def test_empty_std_devs_raises(self) -> None:
        with pytest.raises(ValueError, match="std_dev"):
            GaussianNoise([])

    def test_zero_std_dev_is_passthrough(self) -> None:
        noise = GaussianNoise([0.0], seed=42)
        values = [1.0, 2.0, 3.0]
        noisy = noise.apply(values)
        assert noisy == pytest.approx(values)

    def test_noise_applied_when_nonzero_std(self) -> None:
        noise = GaussianNoise([1.0], seed=42)
        values = [5.0, 5.0, 5.0]
        noisy = noise.apply(values)
        # With seed=42, noise is deterministic and non-zero
        assert noisy != pytest.approx(values, abs=0.0)

    def test_output_length_matches_input(self) -> None:
        noise = GaussianNoise([0.1], seed=0)
        values = [1.0, 2.0, 3.0, 4.0]
        noisy = noise.apply(values)
        assert len(noisy) == len(values)

    def test_last_std_dev_reused_for_extra_dims(self) -> None:
        noise = GaussianNoise([0.0], seed=0)  # Single zero std
        values = [1.0, 2.0, 3.0]  # More dims than std_devs
        noisy = noise.apply(values)
        assert noisy == pytest.approx(values)

    def test_callable_interface(self) -> None:
        noise = GaussianNoise([0.0])
        result = noise([1.0, 2.0])
        assert result == pytest.approx([1.0, 2.0])

    def test_repr(self) -> None:
        noise = GaussianNoise([0.01, 0.02])
        assert "GaussianNoise" in repr(noise)

    def test_nonzero_mean_biases_output(self) -> None:
        noise = GaussianNoise([0.0], mean=5.0, seed=0)
        values = [0.0]
        noisy = noise.apply(values)
        # With std=0, mean is still added via gauss(mean, 0) — but actually
        # gauss(5.0, 0.0) raises in Python. The code checks std != 0.0, so
        # mean bias only applies when std > 0. Let's use a tiny std.
        noise2 = GaussianNoise([0.0001], mean=5.0, seed=0)
        noisy2 = noise2.apply([0.0])
        assert noisy2[0] > 4.0  # Biased toward mean


class TestUniformNoise:
    def test_empty_magnitudes_raises(self) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            UniformNoise([])

    def test_zero_magnitude_is_passthrough(self) -> None:
        noise = UniformNoise([0.0], seed=42)
        values = [1.0, 2.0, 3.0]
        noisy = noise.apply(values)
        assert noisy == pytest.approx(values)

    def test_noise_within_bounds(self) -> None:
        magnitude = 0.5
        noise = UniformNoise([magnitude], seed=0)
        values = [10.0] * 100
        for result in [noise.apply(values)]:
            for orig, noisy_val in zip(values, result):
                assert abs(noisy_val - orig) <= magnitude + 1e-9

    def test_output_length_matches_input(self) -> None:
        noise = UniformNoise([0.1], seed=0)
        assert len(noise.apply([1.0, 2.0, 3.0])) == 3

    def test_last_magnitude_reused(self) -> None:
        noise = UniformNoise([0.0], seed=0)
        noisy = noise.apply([1.0, 2.0, 3.0])
        assert noisy == pytest.approx([1.0, 2.0, 3.0])

    def test_callable_interface(self) -> None:
        noise = UniformNoise([0.0])
        assert noise([1.0]) == pytest.approx([1.0])

    def test_repr(self) -> None:
        noise = UniformNoise([0.01])
        assert "UniformNoise" in repr(noise)


class TestCompositeNoise:
    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="model"):
            CompositeNoise([])

    def test_single_model_passthrough(self) -> None:
        noise = CompositeNoise([GaussianNoise([0.0])])
        values = [1.0, 2.0]
        assert noise.apply(values) == pytest.approx(values)

    def test_multiple_models_applied_in_sequence(self) -> None:
        g = GaussianNoise([0.0])
        u = UniformNoise([0.0])
        composite = CompositeNoise([g, u])
        values = [3.0]
        result = composite.apply(values)
        assert result == pytest.approx(values)

    def test_callable_interface(self) -> None:
        noise = CompositeNoise([GaussianNoise([0.0])])
        assert noise([5.0]) == pytest.approx([5.0])

    def test_repr(self) -> None:
        noise = CompositeNoise([GaussianNoise([0.01])])
        assert "CompositeNoise" in repr(noise)
