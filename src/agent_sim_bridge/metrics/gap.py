"""SimRealGap — quantify the sim-to-real discrepancy.

The sim-to-real gap is the collection of measurable differences between
paired trajectories collected in simulation and on real hardware under the
same policy and conditions.  Closing this gap is the primary goal of domain
randomization and calibration.

GapReport
---------
A :class:`GapReport` is the structured output of one gap measurement.  It
contains per-metric values and an overall gap score (lower is better).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Results of a sim-to-real gap measurement.

    Attributes
    ----------
    observation_mae:
        Mean absolute error between sim and real observations (per element).
    observation_rmse:
        Root mean squared error between sim and real observations.
    reward_mae:
        Mean absolute error between sim and real reward signals.
    reward_bias:
        Signed mean difference (sim - real) — positive means sim over-estimates.
    trajectory_length_ratio:
        Real trajectory length divided by sim trajectory length.  1.0 is ideal.
    overall_gap_score:
        A single summary score combining all sub-metrics.  Computed
        automatically in :meth:`compute_overall_score`.  Lower is better.
    n_sim_steps:
        Number of steps in the simulation trajectory.
    n_real_steps:
        Number of steps in the real trajectory.
    metadata:
        Arbitrary annotations (e.g., policy name, environment version).
    """

    observation_mae: float = 0.0
    observation_rmse: float = 0.0
    reward_mae: float = 0.0
    reward_bias: float = 0.0
    trajectory_length_ratio: float = 1.0
    overall_gap_score: float = 0.0
    n_sim_steps: int = 0
    n_real_steps: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, float | int | object]:
        """Return a flat dict of all gap metrics."""
        return {
            "observation_mae": self.observation_mae,
            "observation_rmse": self.observation_rmse,
            "reward_mae": self.reward_mae,
            "reward_bias": self.reward_bias,
            "trajectory_length_ratio": self.trajectory_length_ratio,
            "overall_gap_score": self.overall_gap_score,
            "n_sim_steps": self.n_sim_steps,
            "n_real_steps": self.n_real_steps,
        }

    def compute_overall_score(
        self,
        obs_weight: float = 0.5,
        reward_weight: float = 0.3,
        length_weight: float = 0.2,
    ) -> float:
        """Compute and store a weighted overall gap score.

        The score is a convex combination of sub-metric scores.  Sub-metric
        contributions are normalised so they are on comparable scales:

        * Observation: MAE (already normalised per element).
        * Reward: MAE.
        * Length: |ratio - 1|.

        Parameters
        ----------
        obs_weight:
            Weight for observation MAE contribution.
        reward_weight:
            Weight for reward MAE contribution.
        length_weight:
            Weight for trajectory length ratio deviation.

        Returns
        -------
        float
            The overall gap score (stored in :attr:`overall_gap_score`).
        """
        total_weight = obs_weight + reward_weight + length_weight
        if total_weight == 0.0:
            self.overall_gap_score = 0.0
            return 0.0

        length_deviation = abs(self.trajectory_length_ratio - 1.0)
        score = (
            obs_weight * self.observation_mae
            + reward_weight * self.reward_mae
            + length_weight * length_deviation
        ) / total_weight
        self.overall_gap_score = score
        return score


class SimRealGap:
    """Measure the sim-to-real gap between paired observation and reward sequences.

    Accepts raw observation lists (each a list of floats) and reward lists,
    then computes element-wise error statistics without external dependencies.

    Parameters
    ----------
    metadata:
        Arbitrary annotations to embed in the :class:`GapReport`.

    Example
    -------
    ::

        gap_metric = SimRealGap(metadata={"policy": "ppo-v2"})
        report = gap_metric.measure_gap(sim_obs, real_obs, sim_rewards, real_rewards)
        print(report.summary())
    """

    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        self._metadata = metadata or {}

    def measure_gap(
        self,
        sim_observations: list[list[float]],
        real_observations: list[list[float]],
        sim_rewards: list[float] | None = None,
        real_rewards: list[float] | None = None,
    ) -> GapReport:
        """Compute the sim-to-real gap from paired trajectories.

        Parameters
        ----------
        sim_observations:
            Observation vectors from the simulation trajectory.
        real_observations:
            Observation vectors from the real trajectory.  Must have the same
            inner dimensionality as ``sim_observations``.
        sim_rewards:
            Scalar rewards from the simulation (optional).
        real_rewards:
            Scalar rewards from the real system (optional).

        Returns
        -------
        GapReport
            Computed gap metrics with :attr:`~GapReport.overall_gap_score`
            already calculated.
        """
        n_sim = len(sim_observations)
        n_real = len(real_observations)
        n_matched = min(n_sim, n_real)

        report = GapReport(
            n_sim_steps=n_sim,
            n_real_steps=n_real,
            metadata=dict(self._metadata),
        )

        if n_matched == 0:
            logger.warning("No overlapping steps to compare; returning zero-gap report.")
            report.compute_overall_score()
            return report

        # Observation statistics over matched steps.
        obs_abs_errors: list[float] = []
        obs_sq_errors: list[float] = []
        for sim_obs, real_obs in zip(sim_observations[:n_matched], real_observations[:n_matched]):
            n_dims = min(len(sim_obs), len(real_obs))
            for dim in range(n_dims):
                diff = sim_obs[dim] - real_obs[dim]
                obs_abs_errors.append(abs(diff))
                obs_sq_errors.append(diff * diff)

        if obs_abs_errors:
            report.observation_mae = sum(obs_abs_errors) / len(obs_abs_errors)
            mean_sq = sum(obs_sq_errors) / len(obs_sq_errors)
            report.observation_rmse = math.sqrt(mean_sq)

        # Reward statistics.
        if sim_rewards is not None and real_rewards is not None:
            sim_rew_matched = sim_rewards[:n_matched]
            real_rew_matched = real_rewards[:n_matched]
            n_reward = min(len(sim_rew_matched), len(real_rew_matched))
            if n_reward > 0:
                diffs = [sim_rew_matched[i] - real_rew_matched[i] for i in range(n_reward)]
                report.reward_mae = sum(abs(d) for d in diffs) / n_reward
                report.reward_bias = sum(diffs) / n_reward

        # Trajectory length ratio.
        if n_sim > 0:
            report.trajectory_length_ratio = n_real / n_sim

        report.compute_overall_score()
        logger.info(
            "Gap analysis complete: obs_mae=%.6f, obs_rmse=%.6f, "
            "reward_mae=%.6f, gap_score=%.6f",
            report.observation_mae,
            report.observation_rmse,
            report.reward_mae,
            report.overall_gap_score,
        )
        return report

    def __repr__(self) -> str:
        return f"SimRealGap(metadata={self._metadata!r})"
