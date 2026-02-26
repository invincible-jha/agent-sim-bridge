"""SimulationSandbox — safe simulation execution with resource limits.

Runs a simulation episode inside a callable policy, enforcing:
* Wall-clock timeout per episode
* Maximum episode step count
* Optional trajectory recording for later analysis

The sandbox is intentionally *not* a subprocess boundary (that would
require pickling environments), but it does track elapsed time and can
interrupt a runaway policy by raising :class:`SandboxTimeoutError`.
"""
from __future__ import annotations

import logging
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from agent_sim_bridge.environment.base import Environment
from agent_sim_bridge.simulation.recorder import TrajectoryRecorder

logger = logging.getLogger(__name__)


class SandboxTimeoutError(RuntimeError):
    """Raised when episode wall-clock time exceeds the configured limit."""


class SandboxMemoryError(RuntimeError):
    """Raised when simulated memory usage exceeds configured limits."""


PolicyCallable = Callable[[NDArray[np.float32]], NDArray[np.float32]]
"""Type alias: a callable that maps an observation to an action."""


@dataclass
class ExecutionResult:
    """Result of a :class:`SimulationSandbox` episode run.

    Attributes
    ----------
    total_reward:
        Cumulative reward over the episode.
    steps:
        Number of timesteps completed.
    terminated:
        Whether the episode ended naturally (terminal state).
    truncated:
        Whether the episode ended due to step/time limit.
    wall_time_seconds:
        Elapsed wall-clock time for the episode.
    recorder:
        :class:`TrajectoryRecorder` populated when ``record=True``.
        ``None`` otherwise.
    error:
        Exception message if the episode aborted unexpectedly.
    """

    total_reward: float = 0.0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    wall_time_seconds: float = 0.0
    recorder: TrajectoryRecorder | None = None
    error: str | None = None
    episode_info: dict[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True when the episode completed without error."""
        return self.error is None

    def summary(self) -> dict[str, object]:
        """Return a plain-dict summary of key metrics."""
        return {
            "total_reward": self.total_reward,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "wall_time_seconds": self.wall_time_seconds,
            "success": self.success,
            "error": self.error,
        }


@contextmanager
def _timeout_context(seconds: float) -> "Iterator[None]":  # type: ignore[type-arg]
    """Context manager that raises SandboxTimeoutError after *seconds*.

    Uses ``signal.SIGALRM`` on POSIX systems.  On Windows (no SIGALRM),
    the timeout is enforced only via manual wall-clock checks between steps.
    """
    import sys

    if seconds <= 0:
        yield
        return

    if sys.platform != "win32":
        def _handler(signum: int, frame: object) -> None:  # noqa: ARG001
            raise SandboxTimeoutError(
                f"Simulation episode exceeded wall-clock timeout of {seconds:.1f}s."
            )

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # On Windows fall through; per-step wall-clock check is the guard.
        yield


# Iterator type annotation
from collections.abc import Iterator  # noqa: E402


class SimulationSandbox:
    """Execute a simulation episode safely with resource enforcement.

    Parameters
    ----------
    environment:
        The simulation environment to run.
    max_steps:
        Hard cap on episode length.
    timeout_seconds:
        Maximum wall-clock time for the entire episode.  On POSIX systems
        this uses ``SIGALRM``; on Windows it is checked between steps.
        Pass ``None`` or 0 to disable.
    record:
        When True a :class:`TrajectoryRecorder` is populated and attached
        to the :class:`ExecutionResult`.
    reset_seed:
        Passed to ``env.reset()`` for reproducible episodes.

    Usage
    -----
    ::

        sandbox = SimulationSandbox(env, max_steps=500, timeout_seconds=30.0)
        result = sandbox.run(policy=my_agent.act)
        print(result.total_reward)
    """

    def __init__(
        self,
        environment: Environment,
        max_steps: int = 1000,
        timeout_seconds: float | None = 60.0,
        record: bool = False,
        reset_seed: int | None = None,
    ) -> None:
        self._env = environment
        self._max_steps = max_steps
        self._timeout = timeout_seconds or 0.0
        self._record = record
        self._reset_seed = reset_seed

    def run(self, policy: PolicyCallable) -> ExecutionResult:
        """Execute a full episode using *policy* to select actions.

        Parameters
        ----------
        policy:
            Callable ``(observation) -> action``.

        Returns
        -------
        ExecutionResult
            Episode statistics and optional trajectory recorder.
        """
        recorder = TrajectoryRecorder() if self._record else None
        result = ExecutionResult(recorder=recorder)
        t_start = time.monotonic()

        try:
            with _timeout_context(self._timeout):
                obs, _ = self._env.reset(seed=self._reset_seed)
                total_reward = 0.0
                steps = 0

                for _ in range(self._max_steps):
                    # Per-step wall-clock guard (especially for Windows)
                    if self._timeout > 0.0:
                        elapsed = time.monotonic() - t_start
                        if elapsed > self._timeout:
                            raise SandboxTimeoutError(
                                f"Simulation episode exceeded timeout of {self._timeout:.1f}s "
                                f"after {steps} steps."
                            )

                    action = policy(obs)
                    step_result = self._env.step(action)
                    total_reward += step_result.reward
                    steps += 1

                    if recorder is not None:
                        recorder.record(
                            observation=obs,
                            action=action,
                            reward=step_result.reward,
                            next_observation=step_result.observation,
                            terminated=step_result.terminated,
                            truncated=step_result.truncated,
                            info=step_result.info,
                        )

                    obs = step_result.observation
                    if step_result.terminated or step_result.truncated:
                        result.terminated = step_result.terminated
                        result.truncated = step_result.truncated
                        break
                else:
                    # Exhausted max_steps without natural termination
                    result.truncated = True

                result.total_reward = total_reward
                result.steps = steps

        except SandboxTimeoutError as exc:
            result.error = str(exc)
            logger.warning("Sandbox timeout: %s", exc)
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            logger.exception("Sandbox episode aborted with error: %s", exc)

        result.wall_time_seconds = time.monotonic() - t_start
        logger.info(
            "Sandbox run complete: steps=%d reward=%.4f wall_time=%.3fs error=%s",
            result.steps,
            result.total_reward,
            result.wall_time_seconds,
            result.error,
        )
        return result
