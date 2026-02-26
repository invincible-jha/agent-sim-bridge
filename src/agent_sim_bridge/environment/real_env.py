"""RealityEnvironment — wraps real-world system interfaces behind the Environment ABC.

Physical hardware (robots, IoT systems, etc.) communicates through a
:class:`RealSystemInterface` protocol object.  The wrapper adds safety
guards: it validates that no action is sent before the environment is
reset, and exposes a configurable step timeout to detect hardware hangs.
"""
from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from agent_sim_bridge.environment.base import (
    Environment,
    EnvironmentInfo,
    SpaceSpec,
    StepResult,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class RealSystemInterface(Protocol):
    """Structural protocol that hardware interface drivers must satisfy."""

    def system_reset(
        self,
        options: dict[str, object] | None,
    ) -> NDArray[np.float32]:
        """Initialise / home the system and return the initial observation."""
        ...

    def system_step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        """Send the action and return ``(obs, reward, terminated, truncated, info)``."""
        ...

    def system_observe(self) -> NDArray[np.float32]:
        """Read sensors without sending a command."""
        ...

    def system_act(self, action: NDArray[np.float32]) -> None:
        """Send a command without reading reward."""
        ...

    def system_close(self) -> None:
        """Gracefully shut down hardware connections."""
        ...

    @property
    def system_state_space(self) -> SpaceSpec:
        """Describe the observation space of the real system."""
        ...

    @property
    def system_action_space(self) -> SpaceSpec:
        """Describe the action space of the real system."""
        ...


class StepTimeoutError(RuntimeError):
    """Raised when a real-system step exceeds the configured timeout."""


class RealityEnvironment(Environment):
    """Real-world environment wrapper that enforces safety guards.

    Parameters
    ----------
    interface:
        An object satisfying the :class:`RealSystemInterface` protocol.
    name:
        Human-readable identifier.
    max_episode_steps:
        Episode truncation limit.
    step_timeout_seconds:
        Maximum seconds to wait for a hardware step response before
        raising :class:`StepTimeoutError`.  Pass ``None`` to disable.
    require_reset:
        When True (default) an action cannot be sent before the first
        ``reset()`` call, preventing accidental actuation on startup.

    Attributes
    ----------
    episode_step:
        Step count within the current episode.
    """

    def __init__(
        self,
        interface: RealSystemInterface,
        name: str = "real-environment",
        max_episode_steps: int = 500,
        step_timeout_seconds: float | None = 5.0,
        require_reset: bool = True,
    ) -> None:
        if not isinstance(interface, RealSystemInterface):
            raise TypeError(
                f"interface must satisfy RealSystemInterface protocol, got {type(interface)!r}"
            )
        self._interface = interface
        self._info = EnvironmentInfo(
            name=name,
            is_simulation=False,
            max_episode_steps=max_episode_steps,
        )
        self._step_timeout = step_timeout_seconds
        self._require_reset = require_reset
        self._has_reset: bool = False
        self._episode_step: int = 0
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def info(self) -> EnvironmentInfo:
        return self._info

    @property
    def state_space(self) -> SpaceSpec:
        return self._interface.system_state_space

    @property
    def action_space(self) -> SpaceSpec:
        return self._interface.system_action_space

    @property
    def episode_step(self) -> int:
        return self._episode_step

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        self._ensure_open()
        if seed is not None:
            logger.warning(
                "%s is a real environment; seed=%s is ignored.", self._info.name, seed
            )
        self._episode_step = 0
        self._has_reset = True
        obs = self._interface.system_reset(options)
        logger.debug("%s reset", self._info.name)
        return obs, {}

    def step(self, action: NDArray[np.float32]) -> StepResult:
        self._ensure_open()
        self._ensure_reset()

        t_start = time.monotonic()
        obs, reward, terminated, truncated, info = self._interface.system_step(action)
        elapsed = time.monotonic() - t_start

        if self._step_timeout is not None and elapsed > self._step_timeout:
            raise StepTimeoutError(
                f"Hardware step took {elapsed:.3f}s, exceeding timeout "
                f"of {self._step_timeout}s for environment {self._info.name!r}."
            )

        self._episode_step += 1
        if self._episode_step >= self._info.max_episode_steps:
            truncated = True

        logger.debug(
            "%s step %d: reward=%.4f terminated=%s elapsed=%.3fs",
            self._info.name,
            self._episode_step,
            reward,
            terminated,
            elapsed,
        )
        return StepResult(obs, reward, terminated, truncated, info)

    def observe(self) -> NDArray[np.float32]:
        self._ensure_open()
        return self._interface.system_observe()

    def act(self, action: NDArray[np.float32]) -> None:
        self._ensure_open()
        self._ensure_reset()
        self._interface.system_act(action)

    def close(self) -> None:
        if not self._closed:
            self._interface.system_close()
            self._closed = True
            logger.debug("%s closed", self._info.name)

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                f"Environment {self._info.name!r} has been closed."
            )

    def _ensure_reset(self) -> None:
        if self._require_reset and not self._has_reset:
            raise RuntimeError(
                f"Environment {self._info.name!r} must be reset before "
                "the first action is sent to prevent accidental actuation."
            )
