"""Unit tests for environment modules.

Covers:
- environment/base.py: SpaceSpec (low/high_array, contains), StepResult,
  EnvironmentInfo, Environment context manager, __repr__, abstract guards
- environment/sim_env.py: SimulationEnvironment full lifecycle, trajectory
  recording, closed-state guard, protocol check
- environment/real_env.py: RealityEnvironment full lifecycle, require_reset
  guard, step timeout, closed-state guard, protocol check
- environment/adapter.py: EnvironmentAdapter with IDENTITY / CLIP / SCALE
  strategies, source property, info forwarding
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent_sim_bridge.environment.base import (
    Environment,
    EnvironmentInfo,
    SpaceSpec,
    StepResult,
)
from agent_sim_bridge.environment.adapter import (
    AdaptationStrategy,
    EnvironmentAdapter,
    _scale_array,
)
from agent_sim_bridge.environment.real_env import (
    RealityEnvironment,
    RealSystemInterface,
    StepTimeoutError,
)
from agent_sim_bridge.environment.sim_env import SimulationEnvironment, SimBackend


# ---------------------------------------------------------------------------
# Helpers — minimal concrete implementations
# ---------------------------------------------------------------------------

_STATE_SPACE = SpaceSpec(shape=(3,), low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0])
_ACTION_SPACE = SpaceSpec(shape=(2,), low=[-1.0, -1.0], high=[1.0, 1.0])


def _make_backend(
    state_shape: tuple[int, ...] = (3,),
    action_shape: tuple[int, ...] = (2,),
    obs_value: float = 0.5,
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
) -> MagicMock:
    """Return a MagicMock that satisfies the SimBackend protocol."""
    backend = MagicMock()
    obs = np.full(state_shape, obs_value, dtype=np.float32)
    state_spec = SpaceSpec(
        shape=state_shape,
        low=[-1.0] * state_shape[0],
        high=[1.0] * state_shape[0],
    )
    action_spec = SpaceSpec(
        shape=action_shape,
        low=[-1.0] * action_shape[0],
        high=[1.0] * action_shape[0],
    )
    backend.backend_state_space = state_spec
    backend.backend_action_space = action_spec
    backend.backend_reset.return_value = obs
    backend.backend_step.return_value = (obs, reward, terminated, truncated, {})
    backend.backend_observe.return_value = obs
    return backend


def _make_interface(
    state_shape: tuple[int, ...] = (3,),
    action_shape: tuple[int, ...] = (2,),
    obs_value: float = 0.5,
) -> MagicMock:
    """Return a MagicMock that satisfies the RealSystemInterface protocol."""
    iface = MagicMock(spec=RealSystemInterface)
    obs = np.full(state_shape, obs_value, dtype=np.float32)
    iface.system_state_space = SpaceSpec(
        shape=state_shape,
        low=[-1.0] * state_shape[0],
        high=[1.0] * state_shape[0],
    )
    iface.system_action_space = SpaceSpec(
        shape=action_shape,
        low=[-1.0] * action_shape[0],
        high=[1.0] * action_shape[0],
    )
    iface.system_reset.return_value = obs
    iface.system_step.return_value = (obs, 1.0, False, False, {})
    iface.system_observe.return_value = obs
    return iface


# ---------------------------------------------------------------------------
# SpaceSpec
# ---------------------------------------------------------------------------


class TestSpaceSpec:
    def test_low_array_explicit(self) -> None:
        spec = SpaceSpec(shape=(2,), low=[-1.0, -2.0], high=[1.0, 2.0])
        np.testing.assert_array_equal(spec.low_array(), [-1.0, -2.0])

    def test_high_array_explicit(self) -> None:
        spec = SpaceSpec(shape=(2,), low=[-1.0, -2.0], high=[1.0, 2.0])
        np.testing.assert_array_equal(spec.high_array(), [1.0, 2.0])

    def test_low_array_none_is_neginf(self) -> None:
        spec = SpaceSpec(shape=(2,))
        assert np.all(np.isneginf(spec.low_array()))

    def test_high_array_none_is_posinf(self) -> None:
        spec = SpaceSpec(shape=(2,))
        assert np.all(np.isposinf(spec.high_array()))

    def test_contains_within_bounds(self) -> None:
        spec = SpaceSpec(shape=(3,), low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0])
        assert spec.contains(np.zeros(3, dtype=np.float32))

    def test_contains_on_boundary(self) -> None:
        spec = SpaceSpec(shape=(2,), low=[-1.0, -1.0], high=[1.0, 1.0])
        assert spec.contains(np.array([1.0, -1.0], dtype=np.float32))

    def test_contains_outside_bounds(self) -> None:
        spec = SpaceSpec(shape=(2,), low=[-1.0, -1.0], high=[1.0, 1.0])
        assert not spec.contains(np.array([1.5, 0.0], dtype=np.float32))

    def test_contains_wrong_shape(self) -> None:
        spec = SpaceSpec(shape=(3,), low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0])
        assert not spec.contains(np.zeros(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_namedtuple_fields(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        result = StepResult(obs, 1.0, False, True, {"key": "val"})
        assert result.reward == 1.0
        assert result.terminated is False
        assert result.truncated is True
        assert result.info["key"] == "val"


# ---------------------------------------------------------------------------
# SimulationEnvironment
# ---------------------------------------------------------------------------


class TestSimulationEnvironment:
    def setup_method(self) -> None:
        self.backend = _make_backend()
        self.env = SimulationEnvironment(self.backend, name="test-sim", max_episode_steps=5)

    def test_info_name(self) -> None:
        assert self.env.info.name == "test-sim"

    def test_info_is_simulation(self) -> None:
        assert self.env.info.is_simulation is True

    def test_state_space_delegates(self) -> None:
        assert self.env.state_space == self.backend.backend_state_space

    def test_action_space_delegates(self) -> None:
        assert self.env.action_space == self.backend.backend_action_space

    def test_reset_returns_obs_and_info(self) -> None:
        obs, info = self.env.reset(seed=42)
        assert obs.shape == (3,)
        assert isinstance(info, dict)
        self.backend.backend_reset.assert_called_once_with(42, None)

    def test_reset_clears_trajectory(self) -> None:
        self.env.reset()
        action = np.zeros(2, dtype=np.float32)
        self.env.step(action)
        self.env.reset()
        assert self.env.episode_step == 0

    def test_step_increments_counter(self) -> None:
        self.env.reset()
        action = np.zeros(2, dtype=np.float32)
        self.env.step(action)
        assert self.env.episode_step == 1

    def test_step_truncates_at_max_steps(self) -> None:
        self.backend.backend_step.return_value = (
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )
        self.env.reset()
        action = np.zeros(2, dtype=np.float32)
        for _ in range(5):
            result = self.env.step(action)
        assert result.truncated is True

    def test_observe_delegates(self) -> None:
        self.env.reset()
        obs = self.env.observe()
        self.backend.backend_observe.assert_called_once()
        assert obs.shape == (3,)

    def test_act_delegates(self) -> None:
        self.env.reset()
        action = np.ones(2, dtype=np.float32)
        self.env.act(action)
        self.backend.backend_act.assert_called_once()

    def test_close_calls_backend_close(self) -> None:
        self.env.close()
        self.backend.backend_close.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        self.env.close()
        self.env.close()
        self.backend.backend_close.assert_called_once()

    def test_closed_env_raises_on_reset(self) -> None:
        self.env.close()
        with pytest.raises(RuntimeError, match="closed"):
            self.env.reset()

    def test_closed_env_raises_on_step(self) -> None:
        self.env.close()
        with pytest.raises(RuntimeError):
            self.env.step(np.zeros(2, dtype=np.float32))

    def test_repr(self) -> None:
        result = repr(self.env)
        assert "test-sim" in result

    def test_context_manager(self) -> None:
        with SimulationEnvironment(self.backend, name="ctx") as env:
            obs, _ = env.reset()
            assert obs is not None
        self.backend.backend_close.assert_called()

    def test_invalid_backend_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="SimBackend protocol"):
            SimulationEnvironment("not-a-backend")  # type: ignore[arg-type]

    def test_trajectory_recording_disabled(self) -> None:
        env = SimulationEnvironment(self.backend, record_trajectories=False)
        env.reset()
        env.step(np.zeros(2, dtype=np.float32))
        assert env.trajectory == []

    def test_trajectory_recording_enabled(self) -> None:
        env = SimulationEnvironment(self.backend, record_trajectories=True)
        env.reset()
        env.step(np.zeros(2, dtype=np.float32))
        assert len(env.trajectory) == 1
        obs, action, reward = env.trajectory[0]
        assert reward == 1.0


# ---------------------------------------------------------------------------
# RealityEnvironment
# ---------------------------------------------------------------------------


class TestRealityEnvironment:
    def setup_method(self) -> None:
        self.iface = _make_interface()
        self.env = RealityEnvironment(
            self.iface,
            name="real-test",
            max_episode_steps=5,
            step_timeout_seconds=10.0,
        )

    def test_info_not_simulation(self) -> None:
        assert self.env.info.is_simulation is False

    def test_info_name(self) -> None:
        assert self.env.info.name == "real-test"

    def test_state_space_delegates(self) -> None:
        assert self.env.state_space == self.iface.system_state_space

    def test_action_space_delegates(self) -> None:
        assert self.env.action_space == self.iface.system_action_space

    def test_reset_returns_obs(self) -> None:
        obs, info = self.env.reset()
        assert obs.shape == (3,)

    def test_reset_seed_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            self.env.reset(seed=99)
        assert "seed=99 is ignored" in caplog.text

    def test_reset_without_seed_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            self.env.reset()
        assert "seed" not in caplog.text

    def test_step_after_reset(self) -> None:
        self.env.reset()
        action = np.zeros(2, dtype=np.float32)
        result = self.env.step(action)
        assert result.reward == 1.0
        assert self.env.episode_step == 1

    def test_step_truncates_at_max_steps(self) -> None:
        self.env.reset()
        action = np.zeros(2, dtype=np.float32)
        for _ in range(5):
            result = self.env.step(action)
        assert result.truncated is True

    def test_step_requires_reset(self) -> None:
        env = RealityEnvironment(self.iface, require_reset=True)
        with pytest.raises(RuntimeError, match="reset before"):
            env.step(np.zeros(2, dtype=np.float32))

    def test_step_no_require_reset(self) -> None:
        env = RealityEnvironment(self.iface, require_reset=False)
        result = env.step(np.zeros(2, dtype=np.float32))
        assert result is not None

    def test_act_requires_reset(self) -> None:
        env = RealityEnvironment(self.iface, require_reset=True)
        with pytest.raises(RuntimeError):
            env.act(np.zeros(2, dtype=np.float32))

    def test_act_after_reset(self) -> None:
        self.env.reset()
        self.env.act(np.ones(2, dtype=np.float32))
        self.iface.system_act.assert_called_once()

    def test_observe(self) -> None:
        obs = self.env.observe()
        self.iface.system_observe.assert_called_once()
        assert obs.shape == (3,)

    def test_close_is_idempotent(self) -> None:
        self.env.close()
        self.env.close()
        self.iface.system_close.assert_called_once()

    def test_closed_env_raises(self) -> None:
        self.env.close()
        with pytest.raises(RuntimeError, match="closed"):
            self.env.reset()

    def test_step_timeout_error(self) -> None:
        """Step that exceeds wall-clock timeout must raise StepTimeoutError."""
        env = RealityEnvironment(
            self.iface,
            step_timeout_seconds=0.001,
        )
        env.reset()

        def slow_step(action: np.ndarray) -> tuple:
            time.sleep(0.05)
            return (np.zeros(3, dtype=np.float32), 0.0, False, False, {})

        self.iface.system_step.side_effect = slow_step
        with pytest.raises(StepTimeoutError, match="exceeding timeout"):
            env.step(np.zeros(2, dtype=np.float32))

    def test_invalid_interface_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="RealSystemInterface protocol"):
            RealityEnvironment("not-an-interface")  # type: ignore[arg-type]

    def test_context_manager(self) -> None:
        with RealityEnvironment(self.iface) as env:
            env.reset()
        self.iface.system_close.assert_called()

    def test_episode_step_resets_on_reset(self) -> None:
        self.env.reset()
        self.env.step(np.zeros(2, dtype=np.float32))
        assert self.env.episode_step == 1
        self.env.reset()
        assert self.env.episode_step == 0


# ---------------------------------------------------------------------------
# _scale_array helper
# ---------------------------------------------------------------------------


class TestScaleArray:
    def test_identity_scale(self) -> None:
        value = np.array([0.5, -0.5], dtype=np.float32)
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        result = _scale_array(value, low, high, low, high)
        np.testing.assert_allclose(result, value, atol=1e-6)

    def test_scale_to_different_range(self) -> None:
        value = np.array([0.0], dtype=np.float32)
        src_low = np.array([-1.0], dtype=np.float32)
        src_high = np.array([1.0], dtype=np.float32)
        tgt_low = np.array([0.0], dtype=np.float32)
        tgt_high = np.array([10.0], dtype=np.float32)
        result = _scale_array(value, src_low, src_high, tgt_low, tgt_high)
        np.testing.assert_allclose(result, [5.0], atol=1e-5)

    def test_degenerate_zero_source_range(self) -> None:
        # Source range is zero — should not divide by zero.
        value = np.array([2.0], dtype=np.float32)
        src_low = np.array([2.0], dtype=np.float32)
        src_high = np.array([2.0], dtype=np.float32)
        tgt_low = np.array([0.0], dtype=np.float32)
        tgt_high = np.array([1.0], dtype=np.float32)
        result = _scale_array(value, src_low, src_high, tgt_low, tgt_high)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# EnvironmentAdapter
# ---------------------------------------------------------------------------


def _make_mock_env(
    obs_value: float = 0.5,
    obs_shape: tuple[int, ...] = (2,),
    state_low: float = -1.0,
    state_high: float = 1.0,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> MagicMock:
    env = MagicMock(spec=Environment)
    obs = np.full(obs_shape, obs_value, dtype=np.float32)
    state_spec = SpaceSpec(
        shape=obs_shape,
        low=[state_low] * obs_shape[0],
        high=[state_high] * obs_shape[0],
    )
    action_spec = SpaceSpec(
        shape=obs_shape,
        low=[action_low] * obs_shape[0],
        high=[action_high] * obs_shape[0],
    )
    env.state_space = state_spec
    env.action_space = action_spec
    env_info = EnvironmentInfo(name="mock-env", is_simulation=True)
    env.info = env_info
    env.reset.return_value = (obs, {})
    env.step.return_value = StepResult(obs, 1.0, False, False, {})
    env.observe.return_value = obs
    return env


class TestEnvironmentAdapterIdentity:
    def setup_method(self) -> None:
        self.source = _make_mock_env()
        self.adapter = EnvironmentAdapter(
            self.source,
            obs_strategy=AdaptationStrategy.IDENTITY,
            action_strategy=AdaptationStrategy.IDENTITY,
        )

    def test_info_name_inherits(self) -> None:
        assert "mock-env" in self.adapter.info.name

    def test_info_name_override(self) -> None:
        adapter = EnvironmentAdapter(self.source, name_override="overridden")
        assert adapter.info.name == "overridden"

    def test_state_space_inherited(self) -> None:
        assert self.adapter.state_space == self.source.state_space

    def test_action_space_inherited(self) -> None:
        assert self.adapter.action_space == self.source.action_space

    def test_source_property(self) -> None:
        assert self.adapter.source is self.source

    def test_reset_passes_through_obs(self) -> None:
        obs, info = self.adapter.reset(seed=1)
        np.testing.assert_array_equal(obs, np.full((2,), 0.5, dtype=np.float32))

    def test_step_passes_through(self) -> None:
        result = self.adapter.step(np.zeros(2, dtype=np.float32))
        assert result.reward == 1.0

    def test_observe_passes_through(self) -> None:
        obs = self.adapter.observe()
        assert obs.shape == (2,)

    def test_act_delegates(self) -> None:
        action = np.ones(2, dtype=np.float32)
        self.adapter.act(action)
        self.source.act.assert_called_once()

    def test_close_delegates(self) -> None:
        self.adapter.close()
        self.source.close.assert_called_once()

    def test_adapter_strategy_in_metadata(self) -> None:
        assert self.adapter.info.metadata["adapter_strategy"] == "identity"


class TestEnvironmentAdapterClip:
    def setup_method(self) -> None:
        self.source = _make_mock_env(obs_value=2.0)  # values will be out of target bounds
        target_state = SpaceSpec(shape=(2,), low=[-1.0, -1.0], high=[1.0, 1.0])
        target_action = SpaceSpec(shape=(2,), low=[-1.0, -1.0], high=[1.0, 1.0])
        self.adapter = EnvironmentAdapter(
            self.source,
            target_state_space=target_state,
            target_action_space=target_action,
            obs_strategy=AdaptationStrategy.CLIP,
            action_strategy=AdaptationStrategy.CLIP,
        )

    def test_obs_is_clipped(self) -> None:
        obs, _ = self.adapter.reset()
        assert np.all(obs <= 1.0)
        assert np.all(obs >= -1.0)

    def test_action_clip_applied(self) -> None:
        # Action outside target bounds should be clipped before passing to source.
        big_action = np.array([5.0, -5.0], dtype=np.float32)
        self.adapter.step(big_action)
        called_action = self.source.step.call_args[0][0]
        assert np.all(called_action <= 1.0)
        assert np.all(called_action >= -1.0)


class TestEnvironmentAdapterScale:
    def setup_method(self) -> None:
        # Source lives in [-1, 1]; expose target in [0, 10].
        self.source = _make_mock_env(obs_value=0.0, state_low=-1.0, state_high=1.0)
        target_state = SpaceSpec(shape=(2,), low=[0.0, 0.0], high=[10.0, 10.0])
        target_action = SpaceSpec(shape=(2,), low=[0.0, 0.0], high=[10.0, 10.0])
        self.adapter = EnvironmentAdapter(
            self.source,
            target_state_space=target_state,
            target_action_space=target_action,
            obs_strategy=AdaptationStrategy.SCALE,
            action_strategy=AdaptationStrategy.SCALE,
        )

    def test_obs_scaled_to_target_range(self) -> None:
        # source returns 0.0 which is midpoint of [-1, 1] → maps to midpoint of [0, 10] = 5.
        obs, _ = self.adapter.reset()
        np.testing.assert_allclose(obs, [5.0, 5.0], atol=1e-5)

    def test_action_scaled_back_to_source(self) -> None:
        # Action 5.0 in [0, 10] should map to 0.0 in [-1, 1].
        mid_action = np.array([5.0, 5.0], dtype=np.float32)
        self.adapter.step(mid_action)
        called_action = self.source.step.call_args[0][0]
        np.testing.assert_allclose(called_action, [0.0, 0.0], atol=1e-5)
