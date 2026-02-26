"""Unit tests for simulation modules.

Covers:
- simulation/recorder.py: TrajectoryStep, TrajectoryRecorder (record, save,
  load, to_bytes, clear, iter, len, max_steps, empty save/load)
- simulation/replay.py: TrajectoryReplay (empty recorder, normal replay,
  early stop on termination, reward correlation, divergence metrics)
- simulation/sandbox.py: SimulationSandbox (run with policy, timeout,
  recording, exception in policy, max_steps exhaustion)
- simulation/scenario.py: Scenario (evaluate, to_reset_options),
  ScenarioManager (save, load, delete, list_names, filter_by_tag)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent_sim_bridge.environment.base import SpaceSpec, StepResult
from agent_sim_bridge.simulation.recorder import TrajectoryRecorder, TrajectoryStep
from agent_sim_bridge.simulation.replay import ReplayResult, TrajectoryReplay
from agent_sim_bridge.simulation.sandbox import (
    ExecutionResult,
    SandboxTimeoutError,
    SimulationSandbox,
)
from agent_sim_bridge.simulation.scenario import (
    Scenario,
    ScenarioManager,
    ScenarioOutcomeCriteria,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(index: int = 0, reward: float = 1.0) -> TrajectoryStep:
    obs = np.zeros(3, dtype=np.float32)
    action = np.ones(2, dtype=np.float32)
    return TrajectoryStep(
        step_index=index,
        observation=obs,
        action=action,
        reward=reward,
        next_observation=obs,
        terminated=False,
        truncated=False,
        timestamp=1000.0 + index,
    )


def _make_env_mock(
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    n_dims: int = 3,
) -> MagicMock:
    obs = np.zeros(n_dims, dtype=np.float32)
    env = MagicMock()
    env.reset.return_value = (obs, {})
    env.step.return_value = StepResult(obs, reward, terminated, truncated, {})
    return env


# ---------------------------------------------------------------------------
# TrajectoryStep
# ---------------------------------------------------------------------------


class TestTrajectoryStep:
    def test_to_dict_roundtrip(self) -> None:
        step = _make_step(index=5, reward=2.5)
        d = step.to_dict()
        restored = TrajectoryStep.from_dict(d)
        assert restored.step_index == 5
        assert restored.reward == 2.5
        np.testing.assert_array_equal(restored.observation, step.observation)

    def test_from_dict_with_info(self) -> None:
        step = _make_step()
        d = step.to_dict()
        d["info"] = {"key": "value"}
        restored = TrajectoryStep.from_dict(d)
        assert restored.info["key"] == "value"

    def test_timestamp_set_automatically(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        step = TrajectoryStep(
            step_index=0,
            observation=obs,
            action=obs,
            reward=0.0,
            next_observation=obs,
            terminated=False,
            truncated=False,
        )
        assert step.timestamp > 0.0


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------


class TestTrajectoryRecorder:
    def test_empty_recorder(self) -> None:
        recorder = TrajectoryRecorder()
        assert len(recorder) == 0
        assert recorder.steps == []

    def test_record_one_step(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.zeros(3, dtype=np.float32)
        recorder.record(obs, obs, 1.0, obs, False, False)
        assert len(recorder) == 1

    def test_record_increments_step_index(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.zeros(3, dtype=np.float32)
        for i in range(3):
            recorder.record(obs, obs, float(i), obs, False, False)
        assert recorder.steps[2].step_index == 2

    def test_record_respects_max_steps(self) -> None:
        recorder = TrajectoryRecorder(max_steps=2)
        obs = np.zeros(3, dtype=np.float32)
        for _ in range(5):
            recorder.record(obs, obs, 1.0, obs, False, False)
        assert len(recorder) == 2

    def test_clear(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.zeros(3, dtype=np.float32)
        recorder.record(obs, obs, 1.0, obs, False, False)
        recorder.clear()
        assert len(recorder) == 0

    def test_iter(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.zeros(3, dtype=np.float32)
        recorder.record(obs, obs, 1.0, obs, False, False)
        steps_from_iter = list(recorder)
        assert len(steps_from_iter) == 1

    def test_repr(self) -> None:
        recorder = TrajectoryRecorder(max_steps=50)
        result = repr(recorder)
        assert "50" in result

    def test_save_and_load_roundtrip(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = np.array([0.5, -0.5], dtype=np.float32)
        recorder.record(obs, action, 2.5, obs * 2, True, False, {"extra": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj.npz"
            recorder.save(path)
            loaded = TrajectoryRecorder.load(path)
        assert len(loaded) == 1
        step = loaded.steps[0]
        np.testing.assert_allclose(step.observation, obs)
        np.testing.assert_allclose(step.action, action)
        assert step.reward == pytest.approx(2.5)
        assert step.terminated is True

    def test_save_appends_npz_suffix(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.zeros(2, dtype=np.float32)
        recorder.record(obs, obs, 0.0, obs, False, False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_suffix"
            recorder.save(path)
            assert (Path(tmp) / "no_suffix.npz").exists()

    def test_save_empty_trajectory_no_error(self) -> None:
        recorder = TrajectoryRecorder()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.npz"
            recorder.save(path)  # Should not raise.
            loaded = TrajectoryRecorder.load(path)
        assert len(loaded) == 0

    def test_load_missing_observations_key_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.npz"
            np.savez_compressed(str(path))
            loaded = TrajectoryRecorder.load(path)
        assert len(loaded) == 0

    def test_to_bytes_roundtrip(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.array([0.1, 0.2], dtype=np.float32)
        recorder.record(obs, obs, 0.5, obs, False, True)
        data = recorder.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_to_bytes_empty(self) -> None:
        recorder = TrajectoryRecorder()
        data = recorder.to_bytes()
        assert isinstance(data, bytes)

    def test_record_with_info(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.zeros(2, dtype=np.float32)
        recorder.record(obs, obs, 0.0, obs, False, False, info={"step_type": "normal"})
        assert recorder.steps[0].info["step_type"] == "normal"

    def test_steps_property_returns_copy(self) -> None:
        recorder = TrajectoryRecorder()
        obs = np.zeros(2, dtype=np.float32)
        recorder.record(obs, obs, 0.0, obs, False, False)
        steps = recorder.steps
        steps.clear()
        assert len(recorder) == 1


# ---------------------------------------------------------------------------
# TrajectoryReplay
# ---------------------------------------------------------------------------


class TestTrajectoryReplay:
    def test_replay_empty_recorder_returns_empty_result(self) -> None:
        env = _make_env_mock()
        recorder = TrajectoryRecorder()
        replayer = TrajectoryReplay(env)
        result = replayer.replay(recorder)
        assert isinstance(result, ReplayResult)
        assert result.original_steps == []
        assert result.replayed_observations == []

    def test_replay_runs_all_steps(self) -> None:
        env = _make_env_mock()
        recorder = TrajectoryRecorder()
        obs = np.zeros(3, dtype=np.float32)
        action = np.ones(2, dtype=np.float32)
        for i in range(3):
            recorder.record(obs, action, 1.0, obs, False, False)
        replayer = TrajectoryReplay(env, reset_seed=42)
        result = replayer.replay(recorder, stop_on_termination=False)
        assert len(result.replayed_observations) == 3
        assert result.observation_mse == pytest.approx(0.0)

    def test_replay_stops_on_termination(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        action = np.ones(2, dtype=np.float32)
        # Step 2 causes termination.
        step_results = [
            StepResult(obs, 1.0, False, False, {}),
            StepResult(obs, 1.0, True, False, {}),  # terminated
            StepResult(obs, 1.0, False, False, {}),
        ]
        env = MagicMock()
        env.reset.return_value = (obs, {})
        env.step.side_effect = step_results

        recorder = TrajectoryRecorder()
        for i in range(3):
            recorder.record(obs, action, 1.0, obs, False, False)

        replayer = TrajectoryReplay(env)
        result = replayer.replay(recorder, stop_on_termination=True)
        # Stops at step index 1 (the second step) — 2 obs collected.
        assert len(result.replayed_observations) == 2

    def test_replay_reward_correlation(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        action = np.ones(2, dtype=np.float32)
        # Same rewards in both original and replayed → perfect correlation.
        rewards = [1.0, 2.0, 3.0]
        step_results = [StepResult(obs, r, False, False, {}) for r in rewards]
        env = MagicMock()
        env.reset.return_value = (obs, {})
        env.step.side_effect = step_results

        recorder = TrajectoryRecorder()
        for r in rewards:
            recorder.record(obs, action, r, obs, False, False)

        replayer = TrajectoryReplay(env)
        result = replayer.replay(recorder, stop_on_termination=False)
        assert result.reward_correlation == pytest.approx(1.0, abs=1e-4)

    def test_replay_single_step_nan_correlation(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        action = np.ones(2, dtype=np.float32)
        env = _make_env_mock()
        recorder = TrajectoryRecorder()
        recorder.record(obs, action, 1.0, obs, False, False)
        replayer = TrajectoryReplay(env)
        result = replayer.replay(recorder, stop_on_termination=False)
        assert np.isnan(result.reward_correlation)

    def test_replay_result_summary(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        action = np.ones(2, dtype=np.float32)
        env = _make_env_mock()
        recorder = TrajectoryRecorder()
        recorder.record(obs, action, 1.0, obs, False, False)
        recorder.record(obs, action, 1.0, obs, False, False)
        replayer = TrajectoryReplay(env)
        result = replayer.replay(recorder, stop_on_termination=False)
        summary = result.summary()
        assert "observation_mse" in summary
        assert "n_steps" in summary

    def test_replay_max_divergence(self) -> None:
        # Source obs is 0, replayed obs is 1 → nonzero divergence.
        orig_obs = np.zeros(3, dtype=np.float32)
        replay_obs = np.ones(3, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        env = MagicMock()
        env.reset.return_value = (replay_obs, {})  # starts at 1
        env.step.return_value = StepResult(replay_obs, 0.0, False, False, {})

        recorder = TrajectoryRecorder()
        recorder.record(orig_obs, action, 0.0, orig_obs, False, False)
        recorder.record(orig_obs, action, 0.0, orig_obs, False, False)

        replayer = TrajectoryReplay(env)
        result = replayer.replay(recorder, stop_on_termination=False)
        assert result.max_obs_divergence > 0.0
        assert result.observation_mse > 0.0


# ---------------------------------------------------------------------------
# SimulationSandbox
# ---------------------------------------------------------------------------


class TestSimulationSandbox:
    def test_run_returns_execution_result(self) -> None:
        env = _make_env_mock(terminated=True)
        sandbox = SimulationSandbox(env, max_steps=10)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert isinstance(result, ExecutionResult)

    def test_run_single_step_terminated(self) -> None:
        env = _make_env_mock(reward=5.0, terminated=True)
        sandbox = SimulationSandbox(env, max_steps=100)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.terminated is True
        assert result.total_reward == pytest.approx(5.0)
        assert result.steps == 1

    def test_run_truncated_at_max_steps(self) -> None:
        env = _make_env_mock(reward=1.0, terminated=False, truncated=False)
        sandbox = SimulationSandbox(env, max_steps=3, timeout_seconds=None)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.truncated is True
        assert result.steps == 3

    def test_run_with_recording_enabled(self) -> None:
        env = _make_env_mock(reward=1.0, terminated=True)
        sandbox = SimulationSandbox(env, max_steps=5, record=True)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.recorder is not None
        assert len(result.recorder) == 1

    def test_run_without_recording(self) -> None:
        env = _make_env_mock(terminated=True)
        sandbox = SimulationSandbox(env, max_steps=5, record=False)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.recorder is None

    def test_run_policy_exception_captured(self) -> None:
        env = _make_env_mock()
        env.reset.return_value = (np.zeros(3, dtype=np.float32), {})

        def bad_policy(obs: np.ndarray) -> np.ndarray:
            raise ValueError("policy exploded")

        sandbox = SimulationSandbox(env, max_steps=5)
        result = sandbox.run(policy=bad_policy)
        assert result.error is not None
        assert "policy exploded" in result.error
        assert not result.success

    def test_run_success_property(self) -> None:
        env = _make_env_mock(terminated=True)
        sandbox = SimulationSandbox(env, max_steps=10)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.success is True

    def test_execution_result_summary(self) -> None:
        env = _make_env_mock(terminated=True)
        sandbox = SimulationSandbox(env, max_steps=5)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        summary = result.summary()
        assert "total_reward" in summary
        assert "steps" in summary
        assert "success" in summary

    def test_wall_time_tracked(self) -> None:
        env = _make_env_mock(terminated=True)
        sandbox = SimulationSandbox(env, max_steps=5)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.wall_time_seconds >= 0.0

    def test_reset_seed_passed(self) -> None:
        env = _make_env_mock(terminated=True)
        sandbox = SimulationSandbox(env, max_steps=5, reset_seed=77)
        sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        env.reset.assert_called_with(seed=77)

    def test_timeout_zero_disables_timeout(self) -> None:
        """timeout_seconds=0 or None must not raise SandboxTimeoutError."""
        env = _make_env_mock(terminated=True)
        sandbox = SimulationSandbox(env, max_steps=5, timeout_seconds=0)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.success

    def test_truncated_flag_propagated(self) -> None:
        env = _make_env_mock(truncated=True)
        sandbox = SimulationSandbox(env, max_steps=10)
        result = sandbox.run(policy=lambda obs: np.zeros(2, dtype=np.float32))
        assert result.truncated is True


# ---------------------------------------------------------------------------
# Scenario / ScenarioManager
# ---------------------------------------------------------------------------


class TestScenario:
    def test_evaluate_pass_all_criteria(self) -> None:
        scenario = Scenario(
            name="test",
            outcome=ScenarioOutcomeCriteria(
                min_reward=5.0,
                max_steps=10,
                require_termination=True,
            ),
        )
        assert scenario.evaluate(total_reward=6.0, steps=8, terminated=True) is True

    def test_evaluate_fails_min_reward(self) -> None:
        scenario = Scenario(
            name="test",
            outcome=ScenarioOutcomeCriteria(min_reward=10.0),
        )
        assert scenario.evaluate(total_reward=5.0, steps=5, terminated=True) is False

    def test_evaluate_fails_max_steps(self) -> None:
        scenario = Scenario(
            name="test",
            outcome=ScenarioOutcomeCriteria(max_steps=5),
        )
        assert scenario.evaluate(total_reward=100.0, steps=10, terminated=True) is False

    def test_evaluate_fails_require_termination(self) -> None:
        scenario = Scenario(
            name="test",
            outcome=ScenarioOutcomeCriteria(require_termination=True),
        )
        assert scenario.evaluate(total_reward=100.0, steps=5, terminated=False) is False

    def test_evaluate_no_criteria_always_passes(self) -> None:
        scenario = Scenario(name="open")
        assert scenario.evaluate(0.0, 0, False) is True

    def test_to_reset_options_empty(self) -> None:
        scenario = Scenario(name="plain")
        options = scenario.to_reset_options()
        assert options == {}

    def test_to_reset_options_with_initial_state(self) -> None:
        scenario = Scenario(
            name="state-override",
            initial_state={"x": 1.5, "y": -0.5},
            env_parameters={"friction": 0.8},
        )
        options = scenario.to_reset_options()
        assert options["initial_state"] == {"x": 1.5, "y": -0.5}
        assert options["friction"] == 0.8


class TestScenarioManager:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager = ScenarioManager(directory=self._tmpdir.name)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def _make_scenario(self, name: str, tags: list[str] | None = None) -> Scenario:
        return Scenario(name=name, description="test", tags=tags or [])

    def test_save_creates_yaml_file(self) -> None:
        scenario = self._make_scenario("alpha")
        path = self.manager.save(scenario)
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_save_returns_correct_path(self) -> None:
        scenario = self._make_scenario("beta")
        path = self.manager.save(scenario)
        assert "beta" in path.name

    def test_load_roundtrip(self) -> None:
        scenario = self._make_scenario("gamma")
        scenario = Scenario(
            name="gamma",
            description="a test scenario",
            seed=42,
            tags=["regression"],
        )
        self.manager.save(scenario)
        loaded = self.manager.load("gamma")
        assert loaded.name == "gamma"
        assert loaded.seed == 42
        assert "regression" in loaded.tags

    def test_load_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            self.manager.load("nonexistent")

    def test_delete_removes_file(self) -> None:
        scenario = self._make_scenario("delta")
        self.manager.save(scenario)
        self.manager.delete("delta")
        with pytest.raises(FileNotFoundError):
            self.manager.load("delta")

    def test_delete_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            self.manager.delete("ghost")

    def test_list_names_empty(self) -> None:
        assert self.manager.list_names() == []

    def test_list_names_with_scenarios(self) -> None:
        self.manager.save(self._make_scenario("a"))
        self.manager.save(self._make_scenario("b"))
        names = self.manager.list_names()
        assert sorted(names) == sorted(["a", "b"])

    def test_list_all_returns_scenario_objects(self) -> None:
        self.manager.save(self._make_scenario("one"))
        all_scenarios = self.manager.list_all()
        assert len(all_scenarios) == 1
        assert isinstance(all_scenarios[0], Scenario)

    def test_filter_by_tag(self) -> None:
        s1 = Scenario(name="s1", tags=["fast", "easy"])
        s2 = Scenario(name="s2", tags=["slow", "hard"])
        s3 = Scenario(name="s3", tags=["fast", "hard"])
        for s in (s1, s2, s3):
            self.manager.save(s)
        fast = self.manager.filter_by_tag("fast")
        assert {s.name for s in fast} == {"s1", "s3"}

    def test_name_with_spaces_saved_correctly(self) -> None:
        scenario = self._make_scenario("my test scenario")
        path = self.manager.save(scenario)
        assert "my_test_scenario" in path.name
        loaded = self.manager.load("my test scenario")
        assert loaded.name == "my test scenario"

    def test_repr(self) -> None:
        result = repr(self.manager)
        assert "ScenarioManager" in result

    def test_list_names_when_directory_missing(self) -> None:
        manager = ScenarioManager(directory="/nonexistent/path/xyz")
        assert manager.list_names() == []
