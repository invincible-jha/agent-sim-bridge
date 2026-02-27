"""Tests for StagingEnvironment and StagingReport.

Covers:
- StagingEnvironment construction (defaults and with chaos)
- add_user / add_tool registration
- run_scenario with sync and async agent functions
- Satisfaction scoring aggregated from users
- Chaos events (partition, latency, degradation)
- Network partition causes scenario to fail
- aggregate_results() produces correct StagingReport
- StagingReport.pass_rate
- StagingReport validation
- repr()
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_sim_bridge.staging.chaos import ChaosConfig
from agent_sim_bridge.staging.environment import StagingEnvironment, StagingReport
from agent_sim_bridge.staging.results import StagingTestResult
from agent_sim_bridge.staging.simulated_tools import SimulatedTool, ToolBehavior
from agent_sim_bridge.staging.simulated_users import UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_profile(
    name: str = "alice",
    persona: str = "technical",
    messages: tuple[str, ...] = ("Hello",),
    expected_outcomes: tuple[str, ...] = ("hello",),
) -> UserProfile:
    return UserProfile(
        name=name,
        persona=persona,
        messages=messages,
        expected_outcomes=expected_outcomes,
    )


def _make_tool_behavior(
    name: str = "search",
    success_rate: float = 1.0,
    latency_ms: float = 0.0,
) -> ToolBehavior:
    return ToolBehavior(name=name, success_rate=success_rate, latency_ms=latency_ms)


async def _echo_agent(
    message: str,
    tools: dict[str, SimulatedTool],
    context: dict[str, Any],
) -> str:
    return f"echo: {message}"


def _sync_echo_agent(
    message: str,
    tools: dict[str, SimulatedTool],
    context: dict[str, Any],
) -> str:
    return f"echo: {message}"


async def _greet_agent(
    message: str,
    tools: dict[str, SimulatedTool],
    context: dict[str, Any],
) -> str:
    return "Hello! I am fine and doing good."


# ---------------------------------------------------------------------------
# StagingEnvironment construction (sync tests)
# ---------------------------------------------------------------------------


class TestStagingEnvironmentConstruction:
    def test_default_construction(self) -> None:
        env = StagingEnvironment()
        assert env.user_count == 0
        assert env.tool_names == []

    def test_with_chaos_config(self) -> None:
        chaos = ChaosConfig(latency_injection_ms=10.0)
        env = StagingEnvironment(chaos_config=chaos)
        assert env.user_count == 0

    def test_add_user_increments_count(self) -> None:
        env = StagingEnvironment()
        env.add_user(_make_user_profile(name="alice"))
        assert env.user_count == 1

    def test_add_multiple_users(self) -> None:
        env = StagingEnvironment()
        env.add_user(_make_user_profile(name="alice"))
        env.add_user(_make_user_profile(name="bob"))
        assert env.user_count == 2

    def test_add_tool_registered(self) -> None:
        env = StagingEnvironment()
        env.add_tool(_make_tool_behavior(name="search"))
        assert "search" in env.tool_names

    def test_add_multiple_tools_sorted(self) -> None:
        env = StagingEnvironment()
        env.add_tool(_make_tool_behavior(name="zebra"))
        env.add_tool(_make_tool_behavior(name="alpha"))
        assert env.tool_names == ["alpha", "zebra"]

    def test_repr_contains_user_and_tool_counts(self) -> None:
        env = StagingEnvironment()
        env.add_user(_make_user_profile())
        representation = repr(env)
        assert "StagingEnvironment" in representation
        assert "users=1" in representation


# ---------------------------------------------------------------------------
# run_scenario (async tests — asyncio_mode=auto)
# ---------------------------------------------------------------------------


class TestRunScenario:
    async def test_basic_scenario_async_agent(self) -> None:
        env = StagingEnvironment()
        env.add_user(_make_user_profile(
            messages=("hello",),
            expected_outcomes=("echo",),
        ))
        result = await env.run_scenario(_echo_agent, "basic-test")
        assert isinstance(result, StagingTestResult)
        assert result.test_name == "basic-test"

    async def test_sync_agent_supported(self) -> None:
        env = StagingEnvironment()
        env.add_user(_make_user_profile(
            messages=("hello",),
            expected_outcomes=("echo",),
        ))
        result = await env.run_scenario(_sync_echo_agent, "sync-test")
        assert isinstance(result, StagingTestResult)

    async def test_passing_scenario_returns_passed_true(self) -> None:
        env = StagingEnvironment(pass_threshold=0.5)
        env.add_user(_make_user_profile(
            messages=("Hi",),
            expected_outcomes=("hello", "fine"),
        ))
        result = await env.run_scenario(_greet_agent, "greet-test")
        assert result.passed is True
        assert result.user_satisfaction >= 0.5

    async def test_no_users_produces_result(self) -> None:
        env = StagingEnvironment()
        result = await env.run_scenario(_echo_agent, "empty-test")
        assert result.test_name == "empty-test"
        assert result.duration_ms >= 0.0

    async def test_result_duration_is_positive(self) -> None:
        env = StagingEnvironment()
        env.add_user(_make_user_profile())
        result = await env.run_scenario(_echo_agent, "timing-test")
        assert result.duration_ms >= 0.0

    async def test_network_partition_causes_fail(self) -> None:
        chaos = ChaosConfig(network_partition_probability=1.0, random_seed=0)
        env = StagingEnvironment(chaos_config=chaos, pass_threshold=0.5)
        env.add_user(_make_user_profile(messages=("hello",)))
        result = await env.run_scenario(_echo_agent, "partition-test")
        assert result.passed is False
        assert any("network_partition" in e for e in result.chaos_events)

    async def test_latency_injection_records_chaos_event(self) -> None:
        chaos = ChaosConfig(latency_injection_ms=1.0, random_seed=0)
        env = StagingEnvironment(chaos_config=chaos)
        env.add_user(_make_user_profile(messages=("hello",)))
        result = await env.run_scenario(_echo_agent, "latency-test")
        assert any("latency_injected" in e for e in result.chaos_events)

    async def test_model_degradation_records_chaos_event(self) -> None:
        chaos = ChaosConfig(model_degradation_rate=0.5, random_seed=0)
        env = StagingEnvironment(chaos_config=chaos)

        async def long_response_agent(message: str, tools: dict, context: dict) -> str:
            return "hello " * 20  # 120 chars — half removed

        env.add_user(_make_user_profile(messages=("test",), expected_outcomes=()))
        result = await env.run_scenario(long_response_agent, "degrade-test")
        assert any("model_output_degraded" in e for e in result.chaos_events)

    async def test_agent_error_recorded_in_errors(self) -> None:
        async def failing_agent(message: str, tools: dict, context: dict) -> str:
            raise ValueError("agent exploded")

        env = StagingEnvironment()
        env.add_user(_make_user_profile(messages=("hello",)))
        result = await env.run_scenario(failing_agent, "error-test")
        assert result.has_errors
        assert any("agent exploded" in e for e in result.errors)

    async def test_multiple_users_satisfaction_aggregated(self) -> None:
        env = StagingEnvironment(pass_threshold=0.5)
        # User 1: expects "echo" — agent response contains it
        env.add_user(_make_user_profile(
            name="user1",
            messages=("msg1",),
            expected_outcomes=("echo",),
        ))
        # User 2: expects "goodbye" — agent response does NOT contain it
        env.add_user(_make_user_profile(
            name="user2",
            messages=("msg2",),
            expected_outcomes=("goodbye",),
        ))
        result = await env.run_scenario(_echo_agent, "multi-user-test")
        # satisfaction should be between 0 and 1
        assert 0.0 <= result.user_satisfaction <= 1.0


# ---------------------------------------------------------------------------
# aggregate_results (sync)
# ---------------------------------------------------------------------------


class TestAggregateResults:
    def _make_result(
        self,
        name: str = "test",
        passed: bool = True,
        satisfaction: float = 0.8,
        duration_ms: float = 100.0,
    ) -> StagingTestResult:
        return StagingTestResult(
            test_name=name,
            passed=passed,
            duration_ms=duration_ms,
            user_satisfaction=satisfaction,
            chaos_events=(),
            errors=(),
        )

    def test_empty_results(self) -> None:
        env = StagingEnvironment()
        report = env.aggregate_results([])
        assert report.total_scenarios == 0
        assert report.pass_rate == 0.0

    def test_all_passed(self) -> None:
        env = StagingEnvironment()
        results = [self._make_result(name=f"t{i}", passed=True) for i in range(3)]
        report = env.aggregate_results(results)
        assert report.passed == 3
        assert report.failed == 0
        assert report.pass_rate == pytest.approx(1.0)

    def test_all_failed(self) -> None:
        env = StagingEnvironment()
        results = [self._make_result(name=f"t{i}", passed=False) for i in range(2)]
        report = env.aggregate_results(results)
        assert report.passed == 0
        assert report.failed == 2
        assert report.pass_rate == pytest.approx(0.0)

    def test_average_satisfaction_computed(self) -> None:
        env = StagingEnvironment()
        results = [
            self._make_result(name="t1", satisfaction=0.8),
            self._make_result(name="t2", satisfaction=0.4),
        ]
        report = env.aggregate_results(results)
        assert report.average_satisfaction == pytest.approx(0.6)

    def test_average_latency_computed(self) -> None:
        env = StagingEnvironment()
        results = [
            self._make_result(name="t1", duration_ms=100.0),
            self._make_result(name="t2", duration_ms=200.0),
        ]
        report = env.aggregate_results(results)
        assert report.average_latency_ms == pytest.approx(150.0)

    def test_chaos_events_totalled(self) -> None:
        env = StagingEnvironment()
        r1 = StagingTestResult(
            test_name="t1", passed=True, duration_ms=10.0,
            user_satisfaction=1.0,
            chaos_events=("partition", "latency"),
            errors=(),
        )
        r2 = StagingTestResult(
            test_name="t2", passed=True, duration_ms=10.0,
            user_satisfaction=1.0,
            chaos_events=("partition",),
            errors=(),
        )
        report = env.aggregate_results([r1, r2])
        assert report.chaos_events_total == 3

    def test_results_tuple_contains_all(self) -> None:
        env = StagingEnvironment()
        results = [self._make_result(name=f"t{i}") for i in range(4)]
        report = env.aggregate_results(results)
        assert len(report.results) == 4


# ---------------------------------------------------------------------------
# StagingReport validation
# ---------------------------------------------------------------------------


class TestStagingReport:
    def test_valid_report(self) -> None:
        report = StagingReport(
            total_scenarios=5,
            passed=3,
            failed=2,
            average_satisfaction=0.7,
            average_latency_ms=120.0,
            chaos_events_total=4,
            results=(),
        )
        assert report.pass_rate == pytest.approx(0.6)

    def test_negative_total_raises(self) -> None:
        with pytest.raises(ValueError, match="total_scenarios"):
            StagingReport(
                total_scenarios=-1,
                passed=0,
                failed=0,
                average_satisfaction=0.0,
                average_latency_ms=0.0,
                chaos_events_total=0,
                results=(),
            )

    def test_passed_plus_failed_exceeds_total_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds total_scenarios"):
            StagingReport(
                total_scenarios=2,
                passed=2,
                failed=2,
                average_satisfaction=0.0,
                average_latency_ms=0.0,
                chaos_events_total=0,
                results=(),
            )
