"""SimulationEnvironment — wraps a simulation backend behind the Environment ABC.

The wrapper delegates all physics calls to a registered backend (PyBullet,
MuJoCo, GenericSimBackend, etc.) while adding trajectory recording hooks
and episode bookkeeping on top.
"""
from __future__ import annotations

import logging
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
class SimBackend(Protocol):
    """Structural protocol that every simulation backend must satisfy.

    Backends are free to add extra methods; this protocol only names the
    minimum surface required by :class:`SimulationEnvironment`.
    """

    def backend_reset(
        self,
        seed: int | None,
        options: dict[str, object] | None,
    ) -> NDArray[np.float32]:
        """Reset the backend and return the initial observation."""
        ...

    def backend_step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        """Advance by one step and return ``(obs, reward, terminated, truncated, info)``."""
        ...

    def backend_observe(self) -> NDArray[np.float32]:
        """Return the current observation without advancing time."""
        ...

    def backend_act(self, action: NDArray[np.float32]) -> None:
        """Apply an action without computing a reward signal."""
        ...

    def backend_close(self) -> None:
        """Free all backend-held resources."""
        ...

    @property
    def backend_state_space(self) -> SpaceSpec:
        """Describe the observation space."""
        ...

    @property
    def backend_action_space(self) -> SpaceSpec:
        """Describe the action space."""
        ...


class SimulationEnvironment(Environment):
    """Simulation environment that delegates to a :class:`SimBackend`.

    Parameters
    ----------
    backend:
        An object satisfying the :class:`SimBackend` protocol.
    name:
        Human-readable name for this environment instance.
    max_episode_steps:
        Episode truncation limit.
    record_trajectories:
        When True the environment accumulates ``(obs, action, reward)``
        tuples in :attr:`trajectory` for later analysis.

    Attributes
    ----------
    trajectory:
        List of ``(observation, action, reward)`` 3-tuples recorded during
        the current episode when ``record_trajectories=True``.
    episode_step:
        Current step count within the running episode.
    """

    def __init__(
        self,
        backend: SimBackend,
        name: str = "sim-environment",
        max_episode_steps: int = 1000,
        record_trajectories: bool = False,
    ) -> None:
        if not isinstance(backend, SimBackend):
            raise TypeError(
                f"backend must satisfy the SimBackend protocol, got {type(backend)!r}"
            )
        self._backend = backend
        self._info = EnvironmentInfo(
            name=name,
            is_simulation=True,
            max_episode_steps=max_episode_steps,
        )
        self._record = record_trajectories
        self._trajectory: list[
            tuple[NDArray[np.float32], NDArray[np.float32], float]
        ] = []
        self._episode_step: int = 0
        self._current_obs: NDArray[np.float32] = np.zeros(
            self._backend.backend_state_space.shape, dtype=np.float32
        )
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def info(self) -> EnvironmentInfo:
        return self._info

    @property
    def state_space(self) -> SpaceSpec:
        return self._backend.backend_state_space

    @property
    def action_space(self) -> SpaceSpec:
        return self._backend.backend_action_space

    @property
    def trajectory(
        self,
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32], float]]:
        """Accumulated ``(obs, action, reward)`` tuples for the current episode."""
        return list(self._trajectory)

    @property
    def episode_step(self) -> int:
        """Step count within the current episode."""
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
        self._trajectory.clear()
        self._episode_step = 0
        obs = self._backend.backend_reset(seed, options)
        self._current_obs = obs
        logger.debug("%s reset (seed=%s)", self._info.name, seed)
        return obs, {}

    def step(self, action: NDArray[np.float32]) -> StepResult:
        self._ensure_open()
        obs, reward, terminated, truncated, info = self._backend.backend_step(action)
        self._episode_step += 1
        self._current_obs = obs

        if self._record:
            self._trajectory.append((self._current_obs.copy(), action.copy(), reward))

        if self._episode_step >= self._info.max_episode_steps:
            truncated = True

        return StepResult(obs, reward, terminated, truncated, info)

    def observe(self) -> NDArray[np.float32]:
        self._ensure_open()
        return self._backend.backend_observe()

    def act(self, action: NDArray[np.float32]) -> None:
        self._ensure_open()
        self._backend.backend_act(action)

    def close(self) -> None:
        if not self._closed:
            self._backend.backend_close()
            self._closed = True
            logger.debug("%s closed", self._info.name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                f"Environment {self._info.name!r} has been closed and cannot be used."
            )
