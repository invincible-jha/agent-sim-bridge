"""Staging test result data model and aggregation helpers.

Provides the :class:`StagingTestResult` immutable value object that
captures the outcome of a single scenario run inside a staging
environment.

Classes
-------
- StagingTestResult   Frozen dataclass for one scenario's results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagingTestResult:
    """Immutable record of a single staging scenario execution.

    Attributes
    ----------
    test_name:
        Unique identifier for the scenario that was run.
    passed:
        Whether the scenario met its success criteria.
    duration_ms:
        Total wall-clock duration of the scenario, in milliseconds.
    user_satisfaction:
        Aggregated user satisfaction score (0.0–1.0) across all
        simulated user interactions in this scenario.
    chaos_events:
        Tuple of short string descriptions for each chaos event that
        occurred during the run (e.g. ``"network_partition"``,
        ``"latency_injected"``).
    errors:
        Tuple of error message strings encountered during the run.
    """

    test_name: str
    passed: bool
    duration_ms: float
    user_satisfaction: float
    chaos_events: tuple[str, ...]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.test_name:
            raise ValueError("StagingTestResult.test_name must not be empty.")
        if self.duration_ms < 0:
            raise ValueError(
                f"duration_ms must be >= 0, got {self.duration_ms}."
            )
        if not (0.0 <= self.user_satisfaction <= 1.0):
            raise ValueError(
                f"user_satisfaction must be in [0.0, 1.0], got {self.user_satisfaction}."
            )

    @property
    def has_errors(self) -> bool:
        """Return True if any errors were recorded during this scenario."""
        return len(self.errors) > 0

    @property
    def chaos_event_count(self) -> int:
        """Return the number of chaos events that occurred."""
        return len(self.chaos_events)


__all__ = ["StagingTestResult"]
