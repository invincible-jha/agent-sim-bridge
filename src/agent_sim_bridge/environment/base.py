"""Abstract base class for simulation and real-world environments.

All environment implementations must subclass ``Environment`` and implement
every abstract method. State observations and actions are represented as
numpy arrays for framework-agnostic interoperability.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class SpaceSpec(BaseModel):
    """Describes a continuous box space (observation or action).

    Attributes
    ----------
    shape:
        Dimensionality tuple, e.g. ``(3,)`` for a 3-vector or ``(84, 84, 3)``
        for an image.
    low:
        Per-dimension lower bounds.  ``None`` means negative infinity.
    high:
        Per-dimension upper bounds.  ``None`` means positive infinity.
    dtype:
        Numpy dtype string, e.g. ``"float32"``.
    """

    shape: tuple[int, ...]
    low: list[float] | None = None
    high: list[float] | None = None
    dtype: str = "float32"

    model_config = {"frozen": True}

    def low_array(self) -> NDArray[np.float32]:
        """Return lower bounds as a numpy array."""
        if self.low is None:
            return np.full(self.shape, -np.inf, dtype=self.dtype)
        return np.array(self.low, dtype=self.dtype).reshape(self.shape)

    def high_array(self) -> NDArray[np.float32]:
        """Return upper bounds as a numpy array."""
        if self.high is None:
            return np.full(self.shape, np.inf, dtype=self.dtype)
        return np.array(self.high, dtype=self.dtype).reshape(self.shape)

    def contains(self, value: NDArray[np.float32]) -> bool:
        """Return True if *value* lies within the space bounds."""
        arr = np.asarray(value, dtype=self.dtype)
        if arr.shape != self.shape:
            return False
        return bool(
            np.all(arr >= self.low_array()) and np.all(arr <= self.high_array())
        )


class StepResult(NamedTuple):
    """Returned by :meth:`Environment.step`.

    Attributes
    ----------
    observation:
        New state observation as a numpy array.
    reward:
        Scalar reward signal.
    terminated:
        True when the episode has ended due to a terminal state.
    truncated:
        True when the episode ended due to time or resource limits.
    info:
        Auxiliary diagnostic information (not used for learning).
    """

    observation: NDArray[np.float32]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class EnvironmentInfo(BaseModel):
    """Metadata describing an environment instance.

    Attributes
    ----------
    name:
        Human-readable identifier.
    version:
        Semantic version string.
    is_simulation:
        True for simulation environments, False for real-world.
    max_episode_steps:
        Maximum allowed steps per episode before truncation.
    metadata:
        Arbitrary key/value metadata.
    """

    name: str
    version: str = "0.1.0"
    is_simulation: bool = True
    max_episode_steps: int = Field(default=1000, gt=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class Environment(ABC):
    """Abstract base for all simulation and real-world environment wrappers.

    Subclasses must implement every abstract method.  The interface is
    intentionally minimal — close enough to Gymnasium so adapters between
    the two are trivial to write.

    Type parameter conventions
    --------------------------
    * Observations are always ``NDArray[np.float32]``.
    * Actions are always ``NDArray[np.float32]``.
    * ``info`` dicts may contain arbitrary serialisable values.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def info(self) -> EnvironmentInfo:
        """Return metadata about this environment."""

    # ------------------------------------------------------------------
    # Space descriptors
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def state_space(self) -> SpaceSpec:
        """Describe the observation (state) space."""

    @property
    @abstractmethod
    def action_space(self) -> SpaceSpec:
        """Describe the action space."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        """Reset the environment to its initial state.

        Parameters
        ----------
        seed:
            Optional RNG seed for reproducibility.
        options:
            Implementation-specific reset options.

        Returns
        -------
        observation:
            Initial state observation.
        info:
            Auxiliary info dict (may be empty).
        """

    @abstractmethod
    def step(
        self,
        action: NDArray[np.float32],
    ) -> StepResult:
        """Advance the environment by one timestep.

        Parameters
        ----------
        action:
            Control input array matching :attr:`action_space`.

        Returns
        -------
        StepResult
            ``(observation, reward, terminated, truncated, info)`` tuple.
        """

    @abstractmethod
    def observe(self) -> NDArray[np.float32]:
        """Return the current observation without advancing the timestep.

        Useful for reading state between steps or after reset.
        """

    @abstractmethod
    def act(self, action: NDArray[np.float32]) -> None:
        """Apply an action without computing reward or checking termination.

        Some physical systems separate actuation from observation reads;
        this method is provided as a lower-level primitive for those cases.

        Parameters
        ----------
        action:
            Control input array matching :attr:`action_space`.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by the environment.

        Must be idempotent — calling ``close`` a second time must not raise.
        """

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Environment":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.info.name!r}, "
            f"is_simulation={self.info.is_simulation})"
        )
