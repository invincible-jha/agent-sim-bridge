"""Validation environment abstractions for sim-to-real harness.

Defines the abstract Environment protocol and value objects for
passing inputs/outputs between sim and real environments, along
with a deterministic MockEnvironment for testing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class EnvironmentInput:
    """Immutable input to an environment execution.

    Attributes
    ----------
    prompt:
        The text prompt or instruction to execute.
    context:
        Arbitrary contextual data accompanying the prompt.
    scenario_id:
        Optional identifier linking this input to a parent scenario.
    """

    prompt: str
    context: dict[str, object] = field(default_factory=dict)
    scenario_id: str = ""


@dataclass(frozen=True)
class EnvironmentOutput:
    """Immutable output produced by an environment execution.

    Attributes
    ----------
    response:
        The textual response produced by the environment.
    latency_seconds:
        Wall-clock time taken to produce the response.
    metadata:
        Arbitrary annotations from the environment.
    timestamp:
        ISO-8601 UTC timestamp of when this output was produced.
    """

    response: str
    latency_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Environment(ABC):
    """Abstract base class for simulation and production environments.

    Both sim and real environments implement this protocol so the
    ValidationHarness can treat them interchangeably.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this environment."""
        ...

    @property
    @abstractmethod
    def environment_type(self) -> str:
        """Either ``'simulation'`` or ``'production'``."""
        ...

    @abstractmethod
    def execute(self, input_data: EnvironmentInput) -> EnvironmentOutput:
        """Execute one input and return the output.

        Parameters
        ----------
        input_data:
            The input to execute in this environment.

        Returns
        -------
        EnvironmentOutput
            The response, latency, and metadata produced.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state held by this environment."""
        ...


class MockEnvironment(Environment):
    """Deterministic mock environment for unit testing.

    Returns pre-configured responses keyed by prompt text.  Falls back to
    a generic response when the prompt is not in the lookup table.

    Parameters
    ----------
    name:
        Human-readable name for this mock instance.
    env_type:
        Either ``'simulation'`` or ``'production'``.
    responses:
        Optional mapping of prompt text to response text.
    fixed_latency:
        Latency (seconds) to report for every execution.
    """

    def __init__(
        self,
        name: str = "mock",
        env_type: str = "simulation",
        responses: dict[str, str] | None = None,
        fixed_latency: float = 0.001,
    ) -> None:
        self._name = name
        self._env_type = env_type
        self._responses: dict[str, str] = responses if responses is not None else {}
        self._fixed_latency = fixed_latency
        self._call_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def environment_type(self) -> str:
        return self._env_type

    @property
    def call_count(self) -> int:
        """Number of times :meth:`execute` has been called since last reset."""
        return self._call_count

    def execute(self, input_data: EnvironmentInput) -> EnvironmentOutput:
        """Return a deterministic response for the given prompt.

        Parameters
        ----------
        input_data:
            The input to process.

        Returns
        -------
        EnvironmentOutput
            Response from the lookup table or a generated fallback.
        """
        self._call_count += 1
        response = self._responses.get(
            input_data.prompt,
            f"Mock response to: {input_data.prompt}",
        )
        return EnvironmentOutput(
            response=response,
            latency_seconds=self._fixed_latency,
        )

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        return (
            f"MockEnvironment(name={self._name!r}, "
            f"env_type={self._env_type!r}, "
            f"call_count={self._call_count})"
        )
