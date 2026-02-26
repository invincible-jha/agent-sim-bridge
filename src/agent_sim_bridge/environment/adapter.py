"""EnvironmentAdapter — adapt between sim and real environment interfaces.

The adapter normalises observation and action spaces so agent code written
against a simulation environment can be deployed against a real one without
modification, and vice versa.

Adaptation strategies
---------------------
* ``clip`` — clip observations/actions to the target space bounds.
* ``scale`` — linearly scale from source space bounds to target bounds.
* ``identity`` — pass through unchanged (default; useful when spaces already match).
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from agent_sim_bridge.environment.base import (
    Environment,
    EnvironmentInfo,
    SpaceSpec,
    StepResult,
)

logger = logging.getLogger(__name__)


class AdaptationStrategy(str, Enum):
    """Supported observation/action normalisation strategies."""

    IDENTITY = "identity"
    CLIP = "clip"
    SCALE = "scale"


def _scale_array(
    value: NDArray[np.float32],
    source_low: NDArray[np.float32],
    source_high: NDArray[np.float32],
    target_low: NDArray[np.float32],
    target_high: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Linearly rescale *value* from source bounds to target bounds."""
    source_range = source_high - source_low
    target_range = target_high - target_low
    # Avoid division by zero: where range is zero, output the target midpoint.
    safe_range = np.where(source_range == 0.0, 1.0, source_range)
    normalised = (value - source_low) / safe_range
    scaled = normalised * target_range + target_low
    return scaled.astype(np.float32)


class EnvironmentAdapter(Environment):
    """Wraps a source environment and adapts its interface to a target space spec.

    Typical use: an agent trained in simulation has a specific observation/
    action space.  When deploying to hardware the actual interface may differ
    slightly.  :class:`EnvironmentAdapter` bridges the gap so the agent code
    does not need to change.

    Parameters
    ----------
    source:
        The underlying environment (sim or real) to wrap.
    target_state_space:
        The observation space the adapter should *present* to callers.
        Pass ``None`` to inherit the source's state space unchanged.
    target_action_space:
        The action space the adapter should *accept* from callers.
        Pass ``None`` to inherit the source's action space unchanged.
    obs_strategy:
        How to transform observations from source → target space.
    action_strategy:
        How to transform actions from target → source space.
    name_override:
        Optional name override for the adapted environment.
    """

    def __init__(
        self,
        source: Environment,
        target_state_space: SpaceSpec | None = None,
        target_action_space: SpaceSpec | None = None,
        obs_strategy: AdaptationStrategy = AdaptationStrategy.IDENTITY,
        action_strategy: AdaptationStrategy = AdaptationStrategy.IDENTITY,
        name_override: str | None = None,
    ) -> None:
        self._source = source
        self._target_state_space = target_state_space or source.state_space
        self._target_action_space = target_action_space or source.action_space
        self._obs_strategy = obs_strategy
        self._action_strategy = action_strategy
        source_info = source.info
        self._info = EnvironmentInfo(
            name=name_override or f"adapted({source_info.name})",
            version=source_info.version,
            is_simulation=source_info.is_simulation,
            max_episode_steps=source_info.max_episode_steps,
            metadata={**source_info.metadata, "adapter_strategy": obs_strategy.value},
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def info(self) -> EnvironmentInfo:
        return self._info

    @property
    def state_space(self) -> SpaceSpec:
        return self._target_state_space

    @property
    def action_space(self) -> SpaceSpec:
        return self._target_action_space

    # ------------------------------------------------------------------
    # Observation transform: source → target
    # ------------------------------------------------------------------

    def _adapt_obs(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._obs_strategy == AdaptationStrategy.IDENTITY:
            return obs
        src_space = self._source.state_space
        tgt_space = self._target_state_space
        if self._obs_strategy == AdaptationStrategy.CLIP:
            return np.clip(obs, tgt_space.low_array(), tgt_space.high_array()).astype(
                np.float32
            )
        # SCALE
        return _scale_array(
            obs,
            src_space.low_array(),
            src_space.high_array(),
            tgt_space.low_array(),
            tgt_space.high_array(),
        )

    # ------------------------------------------------------------------
    # Action transform: target → source
    # ------------------------------------------------------------------

    def _adapt_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._action_strategy == AdaptationStrategy.IDENTITY:
            return action
        src_space = self._source.action_space
        tgt_space = self._target_action_space
        if self._action_strategy == AdaptationStrategy.CLIP:
            return np.clip(action, src_space.low_array(), src_space.high_array()).astype(
                np.float32
            )
        # SCALE: invert target→source
        return _scale_array(
            action,
            tgt_space.low_array(),
            tgt_space.high_array(),
            src_space.low_array(),
            src_space.high_array(),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        obs, info = self._source.reset(seed=seed, options=options)
        return self._adapt_obs(obs), info

    def step(self, action: NDArray[np.float32]) -> StepResult:
        adapted_action = self._adapt_action(action)
        result = self._source.step(adapted_action)
        adapted_obs = self._adapt_obs(result.observation)
        return StepResult(
            adapted_obs,
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def observe(self) -> NDArray[np.float32]:
        return self._adapt_obs(self._source.observe())

    def act(self, action: NDArray[np.float32]) -> None:
        self._source.act(self._adapt_action(action))

    def close(self) -> None:
        self._source.close()

    @property
    def source(self) -> Environment:
        """The underlying (unwrapped) environment."""
        return self._source
