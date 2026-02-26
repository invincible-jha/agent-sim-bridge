"""TrajectoryReplay — replay recorded trajectories in simulation.

Feeds pre-recorded actions back into a simulation environment step-by-step,
collecting new observations for comparison against the originals.  The
divergence between replayed and original trajectories quantifies the
reproducibility of the simulation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from agent_sim_bridge.environment.base import Environment
from agent_sim_bridge.simulation.recorder import TrajectoryRecorder, TrajectoryStep

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Output of a :class:`TrajectoryReplay` run.

    Attributes
    ----------
    original_steps:
        The steps from the source trajectory.
    replayed_observations:
        Observations collected during replay (one per step).
    observation_mse:
        Mean squared error between original and replayed observations.
    max_obs_divergence:
        Maximum per-step L2 norm of observation difference.
    reward_correlation:
        Pearson correlation between original and replayed reward signals.
        ``NaN`` if fewer than 2 steps were replayed.
    """

    original_steps: list[TrajectoryStep]
    replayed_observations: list[NDArray[np.float32]]
    observation_mse: float = 0.0
    max_obs_divergence: float = 0.0
    reward_correlation: float = float("nan")
    replayed_rewards: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Return key metrics as a plain dict."""
        return {
            "observation_mse": self.observation_mse,
            "max_obs_divergence": self.max_obs_divergence,
            "reward_correlation": self.reward_correlation,
            "n_steps": float(len(self.original_steps)),
        }


class TrajectoryReplay:
    """Replays a recorded trajectory through a simulation environment.

    Parameters
    ----------
    environment:
        The simulation environment to replay into.  Must not be closed.
    reset_seed:
        RNG seed passed to ``env.reset()`` at the start of each replay.
        Using the same seed as the original recording improves comparability.

    Usage
    -----
    ::

        recorder = TrajectoryRecorder.load("my_trajectory.npz")
        replayer = TrajectoryReplay(sim_env, reset_seed=42)
        result = replayer.replay(recorder)
        print(result.summary())
    """

    def __init__(
        self,
        environment: Environment,
        reset_seed: int | None = None,
    ) -> None:
        self._env = environment
        self._reset_seed = reset_seed

    def replay(
        self,
        recorder: TrajectoryRecorder,
        stop_on_termination: bool = True,
    ) -> ReplayResult:
        """Execute all recorded actions in the environment and collect results.

        Parameters
        ----------
        recorder:
            The trajectory to replay. Must contain at least one step.
        stop_on_termination:
            If True (default), replay stops early when the environment
            signals termination, even if there are remaining steps.

        Returns
        -------
        ReplayResult
            Observations, divergence metrics, and reward statistics.
        """
        steps = recorder.steps
        if not steps:
            logger.warning("TrajectoryReplay called with empty recorder.")
            return ReplayResult(original_steps=[], replayed_observations=[])

        obs, _ = self._env.reset(seed=self._reset_seed)
        replayed_observations: list[NDArray[np.float32]] = []
        replayed_rewards: list[float] = []

        for step in steps:
            replayed_observations.append(obs.copy())
            result = self._env.step(step.action)
            replayed_rewards.append(result.reward)
            obs = result.observation
            if stop_on_termination and (result.terminated or result.truncated):
                logger.debug(
                    "Replay stopped early at step %d (terminated=%s, truncated=%s)",
                    step.step_index,
                    result.terminated,
                    result.truncated,
                )
                break

        # Compute divergence metrics only over matched steps
        n_matched = len(replayed_observations)
        matched_original = steps[:n_matched]

        orig_obs_arr = np.stack(
            [s.observation for s in matched_original], axis=0
        )
        replay_obs_arr = np.stack(replayed_observations, axis=0)
        diff = orig_obs_arr - replay_obs_arr
        observation_mse = float(np.mean(diff**2))
        step_norms = np.linalg.norm(diff.reshape(n_matched, -1), axis=1)
        max_divergence = float(np.max(step_norms)) if n_matched > 0 else 0.0

        orig_rewards = np.array([s.reward for s in matched_original], dtype=np.float64)
        replay_rewards_arr = np.array(replayed_rewards, dtype=np.float64)
        if n_matched >= 2:
            correlation_matrix = np.corrcoef(orig_rewards, replay_rewards_arr)
            reward_correlation = float(correlation_matrix[0, 1])
        else:
            reward_correlation = float("nan")

        logger.info(
            "Replay complete: %d/%d steps, obs_mse=%.6f, reward_corr=%.4f",
            n_matched,
            len(steps),
            observation_mse,
            reward_correlation,
        )

        return ReplayResult(
            original_steps=matched_original,
            replayed_observations=replayed_observations,
            observation_mse=observation_mse,
            max_obs_divergence=max_divergence,
            reward_correlation=reward_correlation,
            replayed_rewards=replayed_rewards,
        )
