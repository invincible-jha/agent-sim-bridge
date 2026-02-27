"""Chaos injection engine for agent staging environments.

Injects controlled failures into a staging scenario to verify that an
agent handles adverse conditions gracefully:

- **Network partitions**: randomly prevent calls from completing.
- **Latency injection**: add configurable millisecond delays.
- **Model degradation**: corrupt or truncate model outputs by a
  configurable rate.

Classes
-------
- ChaosConfig   Frozen, validated configuration for chaos parameters.
- ChaosEngine   Stateful engine that applies chaos based on config.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosConfig:
    """Immutable chaos configuration for a staging environment.

    Attributes
    ----------
    network_partition_probability:
        Probability (0.0–1.0) that a given call will be blocked as if
        the network is partitioned.  0.0 means no partitions.
    latency_injection_ms:
        Additional milliseconds of simulated latency added to every
        call.  0.0 means no extra latency.
    model_degradation_rate:
        Fraction (0.0–1.0) by which model output quality is degraded.
        At 1.0 the output is fully corrupted; at 0.0 it is unchanged.
    tool_failure_rate:
        Probability (0.0–1.0) that any simulated tool call will fail.
    random_seed:
        Optional seed for the random number generator, enabling
        reproducible chaos scenarios.  ``None`` means non-deterministic.
    """

    network_partition_probability: float = 0.0
    latency_injection_ms: float = 0.0
    model_degradation_rate: float = 0.0
    tool_failure_rate: float = 0.0
    random_seed: int | None = None

    def __post_init__(self) -> None:
        self._validate_probability("network_partition_probability", self.network_partition_probability)
        self._validate_probability("model_degradation_rate", self.model_degradation_rate)
        self._validate_probability("tool_failure_rate", self.tool_failure_rate)
        if self.latency_injection_ms < 0:
            raise ValueError(
                f"latency_injection_ms must be >= 0, got {self.latency_injection_ms}."
            )

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"{name} must be in [0.0, 1.0], got {value}."
            )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ChaosEngine:
    """Applies chaos to staging environment interactions.

    The engine uses a seeded :class:`random.Random` instance so that
    results are reproducible when ``config.random_seed`` is set.

    Parameters
    ----------
    config:
        Immutable chaos configuration.
    """

    def __init__(self, config: ChaosConfig) -> None:
        self._config = config
        self._rng = random.Random(config.random_seed)
        self._partition_count: int = 0
        self._latency_total_ms: float = 0.0
        self._degradation_count: int = 0

    @property
    def config(self) -> ChaosConfig:
        """The immutable chaos configuration."""
        return self._config

    @property
    def partition_count(self) -> int:
        """Total number of partition events triggered so far."""
        return self._partition_count

    @property
    def latency_total_ms(self) -> float:
        """Cumulative simulated latency injected so far (milliseconds)."""
        return self._latency_total_ms

    @property
    def degradation_count(self) -> int:
        """Total number of output degradation events applied so far."""
        return self._degradation_count

    def should_partition(self) -> bool:
        """Return True if this interaction should be dropped as a partition.

        Uses the configured :attr:`ChaosConfig.network_partition_probability`.

        Returns
        -------
        bool
            ``True`` means the call should be treated as if the network
            is unavailable.
        """
        if self._config.network_partition_probability <= 0.0:
            return False
        result = self._rng.random() < self._config.network_partition_probability
        if result:
            self._partition_count += 1
            logger.debug("ChaosEngine: network partition injected.")
        return result

    def inject_latency(self) -> float:
        """Return the configured latency injection amount in milliseconds.

        This is a deterministic value (not randomised) — the full
        ``latency_injection_ms`` is always returned when non-zero.  For
        random latency, wrap this in the caller.

        Returns
        -------
        float
            Latency in milliseconds to add (>= 0.0).
        """
        latency = self._config.latency_injection_ms
        if latency > 0.0:
            self._latency_total_ms += latency
            logger.debug("ChaosEngine: injecting %.2f ms latency.", latency)
        return latency

    def degrade_output(self, output: str) -> str:
        """Degrade *output* according to the configured degradation rate.

        Degradation is simulated by truncating the string: a rate of 0.5
        removes the last 50% of characters; a rate of 1.0 returns an
        empty string.  No random noise is added so that results are
        deterministic for a given input.

        Parameters
        ----------
        output:
            Original model output string.

        Returns
        -------
        str
            Degraded output string.
        """
        rate = self._config.model_degradation_rate
        if rate <= 0.0:
            return output
        keep_fraction = 1.0 - rate
        keep_chars = max(0, int(len(output) * keep_fraction))
        degraded = output[:keep_chars]
        self._degradation_count += 1
        logger.debug(
            "ChaosEngine: degraded output from %d to %d chars.",
            len(output),
            len(degraded),
        )
        return degraded

    def should_fail_tool(self) -> bool:
        """Return True if a simulated tool call should fail.

        Uses the configured :attr:`ChaosConfig.tool_failure_rate`.

        Returns
        -------
        bool
            ``True`` means the tool should raise an error.
        """
        if self._config.tool_failure_rate <= 0.0:
            return False
        return self._rng.random() < self._config.tool_failure_rate

    def reset_counters(self) -> None:
        """Reset cumulative event counters without changing configuration."""
        self._partition_count = 0
        self._latency_total_ms = 0.0
        self._degradation_count = 0

    def __repr__(self) -> str:
        return (
            f"ChaosEngine("
            f"partition_prob={self._config.network_partition_probability}, "
            f"latency_ms={self._config.latency_injection_ms}, "
            f"degradation_rate={self._config.model_degradation_rate}"
            f")"
        )


__all__ = [
    "ChaosConfig",
    "ChaosEngine",
]
