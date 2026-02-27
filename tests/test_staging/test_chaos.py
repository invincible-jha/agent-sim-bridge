"""Tests for the ChaosEngine and ChaosConfig.

Covers:
- ChaosConfig validation (probabilities, latency)
- ChaosEngine.should_partition() respects probability
- ChaosEngine.inject_latency() returns configured value
- ChaosEngine.degrade_output() truncates by rate
- ChaosEngine.should_fail_tool() respects rate
- Counter tracking
- reset_counters()
- repr()
"""
from __future__ import annotations

import pytest

from agent_sim_bridge.staging.chaos import ChaosConfig, ChaosEngine


# ---------------------------------------------------------------------------
# ChaosConfig validation
# ---------------------------------------------------------------------------


class TestChaosConfig:
    def test_defaults_are_zero(self) -> None:
        config = ChaosConfig()
        assert config.network_partition_probability == 0.0
        assert config.latency_injection_ms == 0.0
        assert config.model_degradation_rate == 0.0
        assert config.tool_failure_rate == 0.0
        assert config.random_seed is None

    def test_valid_custom_config(self) -> None:
        config = ChaosConfig(
            network_partition_probability=0.1,
            latency_injection_ms=50.0,
            model_degradation_rate=0.2,
            tool_failure_rate=0.05,
            random_seed=42,
        )
        assert config.network_partition_probability == 0.1
        assert config.latency_injection_ms == 50.0
        assert config.model_degradation_rate == 0.2
        assert config.tool_failure_rate == 0.05
        assert config.random_seed == 42

    def test_partition_probability_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="network_partition_probability"):
            ChaosConfig(network_partition_probability=1.1)

    def test_partition_probability_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="network_partition_probability"):
            ChaosConfig(network_partition_probability=-0.1)

    def test_degradation_rate_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="model_degradation_rate"):
            ChaosConfig(model_degradation_rate=1.5)

    def test_tool_failure_rate_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_failure_rate"):
            ChaosConfig(tool_failure_rate=2.0)

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency_injection_ms"):
            ChaosConfig(latency_injection_ms=-1.0)

    def test_frozen(self) -> None:
        config = ChaosConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.latency_injection_ms = 100.0  # type: ignore[misc]

    def test_boundary_values_valid(self) -> None:
        # 0.0 and 1.0 should both be valid
        config = ChaosConfig(
            network_partition_probability=1.0,
            model_degradation_rate=0.0,
            tool_failure_rate=1.0,
        )
        assert config.network_partition_probability == 1.0


# ---------------------------------------------------------------------------
# ChaosEngine.should_partition()
# ---------------------------------------------------------------------------


class TestChaosEngineShouldPartition:
    def test_zero_probability_never_partitions(self) -> None:
        engine = ChaosEngine(ChaosConfig(network_partition_probability=0.0))
        assert all(not engine.should_partition() for _ in range(100))

    def test_full_probability_always_partitions(self) -> None:
        engine = ChaosEngine(ChaosConfig(network_partition_probability=1.0, random_seed=0))
        assert all(engine.should_partition() for _ in range(20))

    def test_partition_count_increments(self) -> None:
        engine = ChaosEngine(ChaosConfig(network_partition_probability=1.0, random_seed=0))
        for _ in range(5):
            engine.should_partition()
        assert engine.partition_count == 5

    def test_no_partition_when_zero_count_stays_zero(self) -> None:
        engine = ChaosEngine(ChaosConfig(network_partition_probability=0.0))
        engine.should_partition()
        engine.should_partition()
        assert engine.partition_count == 0


# ---------------------------------------------------------------------------
# ChaosEngine.inject_latency()
# ---------------------------------------------------------------------------


class TestChaosEngineInjectLatency:
    def test_zero_latency_returns_zero(self) -> None:
        engine = ChaosEngine(ChaosConfig(latency_injection_ms=0.0))
        assert engine.inject_latency() == 0.0

    def test_configured_latency_returned(self) -> None:
        engine = ChaosEngine(ChaosConfig(latency_injection_ms=75.0))
        assert engine.inject_latency() == 75.0

    def test_latency_total_accumulates(self) -> None:
        engine = ChaosEngine(ChaosConfig(latency_injection_ms=10.0))
        engine.inject_latency()
        engine.inject_latency()
        assert engine.latency_total_ms == 20.0

    def test_zero_latency_total_stays_zero(self) -> None:
        engine = ChaosEngine(ChaosConfig(latency_injection_ms=0.0))
        engine.inject_latency()
        assert engine.latency_total_ms == 0.0


# ---------------------------------------------------------------------------
# ChaosEngine.degrade_output()
# ---------------------------------------------------------------------------


class TestChaosEngineDegradeOutput:
    def test_zero_rate_returns_unchanged(self) -> None:
        engine = ChaosEngine(ChaosConfig(model_degradation_rate=0.0))
        original = "Hello, I can help you with that."
        assert engine.degrade_output(original) == original

    def test_full_rate_returns_empty(self) -> None:
        engine = ChaosEngine(ChaosConfig(model_degradation_rate=1.0))
        result = engine.degrade_output("some output text")
        assert result == ""

    def test_half_rate_halves_length(self) -> None:
        engine = ChaosEngine(ChaosConfig(model_degradation_rate=0.5))
        original = "0123456789"  # 10 chars
        result = engine.degrade_output(original)
        assert len(result) == 5

    def test_degradation_count_increments(self) -> None:
        engine = ChaosEngine(ChaosConfig(model_degradation_rate=0.5))
        engine.degrade_output("hello world")
        engine.degrade_output("another text")
        assert engine.degradation_count == 2

    def test_zero_rate_no_degradation_count(self) -> None:
        engine = ChaosEngine(ChaosConfig(model_degradation_rate=0.0))
        engine.degrade_output("hello")
        assert engine.degradation_count == 0

    def test_empty_string_returns_empty(self) -> None:
        engine = ChaosEngine(ChaosConfig(model_degradation_rate=0.5))
        assert engine.degrade_output("") == ""


# ---------------------------------------------------------------------------
# ChaosEngine.should_fail_tool()
# ---------------------------------------------------------------------------


class TestChaosEngineShouldFailTool:
    def test_zero_rate_never_fails(self) -> None:
        engine = ChaosEngine(ChaosConfig(tool_failure_rate=0.0))
        assert all(not engine.should_fail_tool() for _ in range(50))

    def test_full_rate_always_fails(self) -> None:
        engine = ChaosEngine(ChaosConfig(tool_failure_rate=1.0, random_seed=0))
        assert all(engine.should_fail_tool() for _ in range(20))


# ---------------------------------------------------------------------------
# ChaosEngine.reset_counters()
# ---------------------------------------------------------------------------


class TestChaosEngineReset:
    def test_reset_clears_counters(self) -> None:
        engine = ChaosEngine(
            ChaosConfig(
                network_partition_probability=1.0,
                latency_injection_ms=10.0,
                model_degradation_rate=0.5,
                random_seed=0,
            )
        )
        engine.should_partition()
        engine.inject_latency()
        engine.degrade_output("hello")
        engine.reset_counters()
        assert engine.partition_count == 0
        assert engine.latency_total_ms == 0.0
        assert engine.degradation_count == 0

    def test_repr_contains_config_values(self) -> None:
        engine = ChaosEngine(ChaosConfig(network_partition_probability=0.3, latency_injection_ms=25.0))
        representation = repr(engine)
        assert "ChaosEngine" in representation
        assert "0.3" in representation
