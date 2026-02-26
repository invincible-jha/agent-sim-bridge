"""TrajectoryRecorder — record state-action-reward trajectories.

Records a sequence of :class:`TrajectoryStep` objects produced during an
episode, then serialises them to disk as a compressed numpy archive or
loads them back for analysis and replay.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrajectoryStep(BaseModel):
    """One transition in a recorded trajectory.

    Attributes
    ----------
    step_index:
        Zero-based index within the episode.
    observation:
        State observation *before* the action was taken.
    action:
        Action applied at this step.
    reward:
        Scalar reward received.
    next_observation:
        State observation *after* the action.
    terminated:
        Episode ended due to terminal state.
    truncated:
        Episode ended due to time/resource limit.
    timestamp:
        Wall-clock time of the step (seconds since epoch).
    info:
        Auxiliary diagnostic dictionary.
    """

    model_config = {"arbitrary_types_allowed": True}

    step_index: int = Field(ge=0)
    observation: NDArray[np.float32]
    action: NDArray[np.float32]
    reward: float
    next_observation: NDArray[np.float32]
    terminated: bool
    truncated: bool
    timestamp: float = Field(default_factory=time.time)
    info: dict[str, object] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict with numpy arrays as lists."""
        return {
            "step_index": self.step_index,
            "observation": self.observation.tolist(),
            "action": self.action.tolist(),
            "reward": self.reward,
            "next_observation": self.next_observation.tolist(),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "timestamp": self.timestamp,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TrajectoryStep":
        """Deserialise from a plain dict."""
        return cls(
            step_index=int(data["step_index"]),  # type: ignore[arg-type]
            observation=np.array(data["observation"], dtype=np.float32),
            action=np.array(data["action"], dtype=np.float32),
            reward=float(data["reward"]),  # type: ignore[arg-type]
            next_observation=np.array(data["next_observation"], dtype=np.float32),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            timestamp=float(data["timestamp"]),  # type: ignore[arg-type]
            info=dict(data.get("info", {})),  # type: ignore[arg-type]
        )


class TrajectoryRecorder:
    """Accumulates trajectory steps in memory, then persists them.

    Parameters
    ----------
    max_steps:
        If set, the recorder stops accepting new steps once this limit is
        reached (existing steps are preserved).

    Usage
    -----
    ::

        recorder = TrajectoryRecorder()
        obs, _ = env.reset()
        while True:
            action = agent.act(obs)
            result = env.step(action)
            recorder.record(obs, action, result.reward, result.observation,
                            result.terminated, result.truncated, result.info)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        recorder.save("my_trajectory.npz")
    """

    def __init__(self, max_steps: int | None = None) -> None:
        self._max_steps = max_steps
        self._steps: list[TrajectoryStep] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        observation: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_observation: NDArray[np.float32],
        terminated: bool,
        truncated: bool,
        info: dict[str, object] | None = None,
    ) -> None:
        """Append one transition to the in-memory buffer.

        Silently ignored if ``max_steps`` has already been reached.
        """
        if self._max_steps is not None and len(self._steps) >= self._max_steps:
            return
        step = TrajectoryStep(
            step_index=len(self._steps),
            observation=observation.copy(),
            action=action.copy(),
            reward=reward,
            next_observation=next_observation.copy(),
            terminated=terminated,
            truncated=truncated,
            info=info or {},
        )
        self._steps.append(step)

    def clear(self) -> None:
        """Discard all recorded steps."""
        self._steps.clear()

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def steps(self) -> list[TrajectoryStep]:
        """Read-only view of recorded steps."""
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[TrajectoryStep]:
        return iter(self._steps)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trajectory to a compressed numpy archive (``.npz``).

        Parameters
        ----------
        path:
            Destination file path. The ``.npz`` extension is appended if
            not already present.
        """
        destination = Path(path)
        if destination.suffix != ".npz":
            destination = destination.with_suffix(".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not self._steps:
            logger.warning("Saving empty trajectory to %s", destination)

        arrays: dict[str, NDArray[np.float32]] = {}
        if self._steps:
            arrays["observations"] = np.stack(
                [s.observation for s in self._steps], axis=0
            )
            arrays["actions"] = np.stack([s.action for s in self._steps], axis=0)
            arrays["rewards"] = np.array(
                [s.reward for s in self._steps], dtype=np.float32
            )
            arrays["next_observations"] = np.stack(
                [s.next_observation for s in self._steps], axis=0
            )
            arrays["terminated"] = np.array(
                [s.terminated for s in self._steps], dtype=bool
            )
            arrays["truncated"] = np.array(
                [s.truncated for s in self._steps], dtype=bool
            )
            arrays["timestamps"] = np.array(
                [s.timestamp for s in self._steps], dtype=np.float64
            )

        np.savez_compressed(destination, **arrays)
        logger.info("Saved %d steps to %s", len(self._steps), destination)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryRecorder":
        """Load a trajectory previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file.

        Returns
        -------
        TrajectoryRecorder
            A new recorder containing the loaded steps.
        """
        source = Path(path)
        if source.suffix != ".npz":
            source = source.with_suffix(".npz")

        data = np.load(source)
        recorder = cls()

        if "observations" not in data:
            logger.warning("Loaded empty trajectory from %s", source)
            return recorder

        n_steps = int(data["observations"].shape[0])
        for i in range(n_steps):
            step = TrajectoryStep(
                step_index=i,
                observation=data["observations"][i].astype(np.float32),
                action=data["actions"][i].astype(np.float32),
                reward=float(data["rewards"][i]),
                next_observation=data["next_observations"][i].astype(np.float32),
                terminated=bool(data["terminated"][i]),
                truncated=bool(data["truncated"][i]),
                timestamp=float(data["timestamps"][i]),
            )
            recorder._steps.append(step)

        logger.info("Loaded %d steps from %s", n_steps, source)
        return recorder

    def to_bytes(self) -> bytes:
        """Serialise to an in-memory bytes buffer (useful for streaming)."""
        buffer = io.BytesIO()
        arrays: dict[str, NDArray[np.float32]] = {}
        if self._steps:
            arrays["observations"] = np.stack(
                [s.observation for s in self._steps], axis=0
            )
            arrays["actions"] = np.stack([s.action for s in self._steps], axis=0)
            arrays["rewards"] = np.array(
                [s.reward for s in self._steps], dtype=np.float32
            )
            arrays["next_observations"] = np.stack(
                [s.next_observation for s in self._steps], axis=0
            )
            arrays["terminated"] = np.array(
                [s.terminated for s in self._steps], dtype=bool
            )
            arrays["truncated"] = np.array(
                [s.truncated for s in self._steps], dtype=bool
            )
            arrays["timestamps"] = np.array(
                [s.timestamp for s in self._steps], dtype=np.float64
            )
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    def __repr__(self) -> str:
        return f"TrajectoryRecorder(steps={len(self._steps)}, max_steps={self._max_steps})"
