"""Tests for simulated tool behaviour.

Covers:
- ToolBehavior validation (name, success_rate, latency_ms)
- SimulatedTool always succeeds with success_rate=1.0
- SimulatedTool always fails with success_rate=0.0
- SimulatedTool call/success/failure counters
- reset_counters()
- execute() passes kwargs through to success payload
- SimulatedToolError attributes
- repr()
"""
from __future__ import annotations

import pytest

from agent_sim_bridge.staging.simulated_tools import (
    SimulatedTool,
    SimulatedToolError,
    ToolBehavior,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_behavior(
    name: str = "search",
    success_rate: float = 1.0,
    latency_ms: float = 0.0,
    error_message: str = "Tool execution failed",
) -> ToolBehavior:
    return ToolBehavior(
        name=name,
        success_rate=success_rate,
        latency_ms=latency_ms,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# ToolBehavior validation
# ---------------------------------------------------------------------------


class TestToolBehavior:
    def test_valid_behavior(self) -> None:
        behavior = _make_behavior()
        assert behavior.name == "search"
        assert behavior.success_rate == 1.0
        assert behavior.latency_ms == 0.0

    def test_frozen(self) -> None:
        behavior = _make_behavior()
        with pytest.raises((AttributeError, TypeError)):
            behavior.name = "other"  # type: ignore[misc]

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ToolBehavior(name="", success_rate=1.0, latency_ms=0.0)

    def test_success_rate_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="success_rate"):
            ToolBehavior(name="tool", success_rate=1.1, latency_ms=0.0)

    def test_success_rate_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="success_rate"):
            ToolBehavior(name="tool", success_rate=-0.1, latency_ms=0.0)

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency_ms"):
            ToolBehavior(name="tool", success_rate=1.0, latency_ms=-5.0)

    def test_boundary_success_rates_valid(self) -> None:
        ToolBehavior(name="t", success_rate=0.0, latency_ms=0.0)
        ToolBehavior(name="t", success_rate=1.0, latency_ms=0.0)


# ---------------------------------------------------------------------------
# SimulatedTool execution — async tests (asyncio_mode=auto)
# ---------------------------------------------------------------------------


class TestSimulatedToolSuccess:
    async def test_always_succeed(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=1.0), random_seed=0)
        result = await tool.execute()
        assert result["status"] == "success"
        assert result["tool"] == "search"

    async def test_kwargs_in_payload(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=1.0), random_seed=0)
        result = await tool.execute(query="hello", limit=10)
        assert result["inputs"]["query"] == "hello"
        assert result["inputs"]["limit"] == 10

    async def test_call_count_increments(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=1.0), random_seed=0)
        await tool.execute()
        await tool.execute()
        assert tool.call_count == 2

    async def test_success_count_increments(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=1.0), random_seed=0)
        await tool.execute()
        await tool.execute()
        assert tool.success_count == 2
        assert tool.failure_count == 0


class TestSimulatedToolFailure:
    async def test_always_fail(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=0.0), random_seed=0)
        with pytest.raises(SimulatedToolError):
            await tool.execute()

    async def test_failure_count_increments(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=0.0), random_seed=0)
        for _ in range(3):
            try:
                await tool.execute()
            except SimulatedToolError:
                pass
        assert tool.failure_count == 3
        assert tool.success_count == 0

    async def test_error_contains_tool_name(self) -> None:
        tool = SimulatedTool(_make_behavior(name="my-tool", success_rate=0.0))
        with pytest.raises(SimulatedToolError) as exc_info:
            await tool.execute()
        assert exc_info.value.tool_name == "my-tool"

    async def test_error_contains_custom_message(self) -> None:
        tool = SimulatedTool(
            _make_behavior(success_rate=0.0, error_message="Custom error"),
        )
        with pytest.raises(SimulatedToolError) as exc_info:
            await tool.execute()
        assert exc_info.value.message == "Custom error"

    async def test_error_call_number_set(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=0.0))
        with pytest.raises(SimulatedToolError) as exc_info:
            await tool.execute()
        assert exc_info.value.call_number == 1


# ---------------------------------------------------------------------------
# reset_counters
# ---------------------------------------------------------------------------


class TestSimulatedToolReset:
    async def test_reset_counters(self) -> None:
        tool = SimulatedTool(_make_behavior(success_rate=1.0), random_seed=0)
        await tool.execute()
        await tool.execute()
        tool.reset_counters()
        assert tool.call_count == 0
        assert tool.success_count == 0
        assert tool.failure_count == 0

    async def test_repr_contains_name_and_call_count(self) -> None:
        tool = SimulatedTool(_make_behavior(name="my-tool", success_rate=1.0), random_seed=0)
        await tool.execute()
        representation = repr(tool)
        assert "my-tool" in representation
        assert "1" in representation
