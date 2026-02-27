"""Simulated tools for agent staging environments.

Provides configurable tool stubs that can simulate success, failure, and
latency so that agents can be tested against realistic tool failure
scenarios without needing real backends.

Classes
-------
- ToolBehavior    Frozen configuration for a simulated tool.
- SimulatedTool   Async callable that behaves according to ToolBehavior.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolBehavior:
    """Immutable configuration for a simulated tool's behaviour.

    Attributes
    ----------
    name:
        Unique identifier for the tool, e.g. ``"web_search"``.
    success_rate:
        Probability (0.0–1.0) that each invocation succeeds.
        At 1.0 the tool always succeeds; at 0.0 it always fails.
    latency_ms:
        Simulated execution latency in milliseconds.  Applied via
        ``asyncio.sleep`` in :meth:`SimulatedTool.execute`.
    error_message:
        Error message text used when the tool fails.
    """

    name: str
    success_rate: float
    latency_ms: float
    error_message: str = "Tool execution failed"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolBehavior.name must not be empty.")
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError(
                f"success_rate must be in [0.0, 1.0], got {self.success_rate}."
            )
        if self.latency_ms < 0:
            raise ValueError(
                f"latency_ms must be >= 0, got {self.latency_ms}."
            )


# ---------------------------------------------------------------------------
# Simulated tool
# ---------------------------------------------------------------------------


class SimulatedTool:
    """Simulates tool behaviour with configurable failures and latency.

    Each call to :meth:`execute` waits for the configured latency and
    then either returns a success payload or raises a
    :class:`SimulatedToolError` based on the configured
    ``success_rate``.

    Parameters
    ----------
    behavior:
        Immutable tool behaviour configuration.
    random_seed:
        Optional RNG seed for reproducible failure patterns.
    """

    def __init__(
        self,
        behavior: ToolBehavior,
        random_seed: int | None = None,
    ) -> None:
        self._behavior = behavior
        self._rng = random.Random(random_seed)
        self._call_count: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0

    @property
    def behavior(self) -> ToolBehavior:
        """The immutable tool behaviour configuration."""
        return self._behavior

    @property
    def name(self) -> str:
        """Tool name (from behavior)."""
        return self._behavior.name

    @property
    def call_count(self) -> int:
        """Total number of times the tool has been invoked."""
        return self._call_count

    @property
    def success_count(self) -> int:
        """Number of successful invocations."""
        return self._success_count

    @property
    def failure_count(self) -> int:
        """Number of failed invocations."""
        return self._failure_count

    async def execute(self, **kwargs: object) -> dict[str, object]:
        """Execute the simulated tool.

        Simulates latency via ``asyncio.sleep`` then either returns a
        success payload or raises :class:`SimulatedToolError`.

        Parameters
        ----------
        **kwargs:
            Arbitrary tool input parameters (passed through to the
            success payload for traceability).

        Returns
        -------
        dict[str, object]
            Success payload containing ``"tool"``, ``"status"``,
            ``"call_count"``, and ``"inputs"`` keys.

        Raises
        ------
        SimulatedToolError
            When the random draw exceeds the configured success rate.
        """
        self._call_count += 1
        call_number = self._call_count

        latency_seconds = self._behavior.latency_ms / 1000.0
        if latency_seconds > 0:
            await asyncio.sleep(latency_seconds)

        if self._rng.random() > self._behavior.success_rate:
            self._failure_count += 1
            logger.debug(
                "SimulatedTool[%s]: call #%d failed.",
                self._behavior.name,
                call_number,
            )
            raise SimulatedToolError(
                tool_name=self._behavior.name,
                message=self._behavior.error_message,
                call_number=call_number,
            )

        self._success_count += 1
        logger.debug(
            "SimulatedTool[%s]: call #%d succeeded.",
            self._behavior.name,
            call_number,
        )
        return {
            "tool": self._behavior.name,
            "status": "success",
            "call_count": call_number,
            "inputs": kwargs,
        }

    def reset_counters(self) -> None:
        """Reset call/success/failure counters without changing behavior."""
        self._call_count = 0
        self._success_count = 0
        self._failure_count = 0

    def __repr__(self) -> str:
        return (
            f"SimulatedTool("
            f"name={self._behavior.name!r}, "
            f"success_rate={self._behavior.success_rate}, "
            f"calls={self._call_count}"
            f")"
        )


class SimulatedToolError(Exception):
    """Raised when a :class:`SimulatedTool` call fails.

    Attributes
    ----------
    tool_name:
        Name of the tool that failed.
    message:
        Configured error message.
    call_number:
        Sequential call number at the time of failure.
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        call_number: int,
    ) -> None:
        self.tool_name = tool_name
        self.message = message
        self.call_number = call_number
        super().__init__(f"[{tool_name}] call #{call_number}: {message}")


__all__ = [
    "ToolBehavior",
    "SimulatedTool",
    "SimulatedToolError",
]
