"""Smoke tests for the 3-line quickstart convenience API.

Covers:
- Simulator class (zero-config run, scenario, outcome, repr)
- quick_recorder — create, record, len, repr
- quick_sandbox — run episode with zero-action policy
- quick_safety_monitor — constraints, violations, emergency stop
- quick_skill_library — create, empty list
- quick_gap_analysis — returns GapReport with expected fields
- quick_transfer_bridge — round-trip sim->real->sim
- Top-level __init__ exports for all convenience symbols
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------


def test_simulator_import() -> None:
    from agent_sim_bridge import Simulator

    sim = Simulator()
    assert sim is not None


def test_simulator_run_returns_result() -> None:
    from agent_sim_bridge import Simulator

    sim = Simulator()
    result = sim.run(steps=5)
    assert result is not None
    assert hasattr(result, "total_reward")
    assert hasattr(result, "steps")


def test_simulator_run_steps_bounded() -> None:
    from agent_sim_bridge import Simulator

    sim = Simulator()
    result = sim.run(steps=10)
    assert result.steps <= 10


def test_simulator_run_total_reward_non_negative() -> None:
    from agent_sim_bridge import Simulator

    sim = Simulator(seed=0)
    result = sim.run(steps=5)
    # Stub backend always gives +1 per step
    assert result.total_reward >= 0.0


def test_simulator_scenario_accessible() -> None:
    from agent_sim_bridge import Simulator
    from agent_sim_bridge.simulation.scenario import Scenario

    sim = Simulator(scenario="navigation-basic")
    assert isinstance(sim.scenario, Scenario)
    assert sim.scenario.name == "navigation-basic"


def test_simulator_scenario_instance_passthrough() -> None:
    from agent_sim_bridge import Simulator
    from agent_sim_bridge.simulation.scenario import Scenario

    custom = Scenario(name="my-scenario", description="test")
    sim = Simulator(scenario=custom)
    assert sim.scenario is custom


def test_simulator_outcome_met_no_criteria() -> None:
    from agent_sim_bridge import Simulator

    sim = Simulator()
    result = sim.run(steps=5)
    # Default scenario has no outcome criteria — every result passes
    assert sim.outcome_met(result) is True


def test_simulator_repr() -> None:
    from agent_sim_bridge import Simulator

    sim = Simulator(scenario="test-run", seed=42)
    text = repr(sim)
    assert "Simulator" in text
    assert "test-run" in text


# ---------------------------------------------------------------------------
# quick_recorder
# ---------------------------------------------------------------------------


def test_quick_recorder_returns_recorder() -> None:
    from agent_sim_bridge import quick_recorder
    from agent_sim_bridge.simulation.recorder import TrajectoryRecorder

    recorder = quick_recorder()
    assert isinstance(recorder, TrajectoryRecorder)


def test_quick_recorder_empty_on_creation() -> None:
    from agent_sim_bridge import quick_recorder

    recorder = quick_recorder()
    assert len(recorder) == 0


def test_quick_recorder_records_step() -> None:
    from agent_sim_bridge import quick_recorder

    recorder = quick_recorder()
    obs = np.zeros(4, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)
    recorder.record(obs, action, 1.0, obs, False, False)
    assert len(recorder) == 1


def test_quick_recorder_respects_max_steps() -> None:
    from agent_sim_bridge import quick_recorder

    recorder = quick_recorder(max_steps=2)
    obs = np.zeros(4, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)
    for _ in range(5):
        recorder.record(obs, action, 1.0, obs, False, False)
    assert len(recorder) == 2


def test_quick_recorder_repr() -> None:
    from agent_sim_bridge import quick_recorder

    recorder = quick_recorder(max_steps=100)
    text = repr(recorder)
    assert "TrajectoryRecorder" in text


# ---------------------------------------------------------------------------
# quick_sandbox
# ---------------------------------------------------------------------------


def _make_stub_env() -> object:
    """Return a ready-to-use SimulationEnvironment backed by the stub."""
    from agent_sim_bridge.convenience import _StubBackend
    from agent_sim_bridge.environment.sim_env import SimulationEnvironment

    return SimulationEnvironment(backend=_StubBackend(seed=0), max_episode_steps=50)


def test_quick_sandbox_returns_execution_result() -> None:
    from agent_sim_bridge import quick_sandbox
    from agent_sim_bridge.simulation.sandbox import ExecutionResult

    env = _make_stub_env()
    result = quick_sandbox(
        env,  # type: ignore[arg-type]
        policy=lambda obs: np.zeros(2, dtype=np.float32),
        max_steps=10,
        timeout_seconds=None,
    )
    assert isinstance(result, ExecutionResult)


def test_quick_sandbox_success_on_clean_episode() -> None:
    from agent_sim_bridge import quick_sandbox

    env = _make_stub_env()
    result = quick_sandbox(
        env,  # type: ignore[arg-type]
        policy=lambda obs: np.zeros(2, dtype=np.float32),
        max_steps=5,
        timeout_seconds=None,
    )
    assert result.success
    assert result.steps <= 5


def test_quick_sandbox_record_flag_attaches_recorder() -> None:
    from agent_sim_bridge import quick_sandbox
    from agent_sim_bridge.simulation.recorder import TrajectoryRecorder

    env = _make_stub_env()
    result = quick_sandbox(
        env,  # type: ignore[arg-type]
        policy=lambda obs: np.zeros(2, dtype=np.float32),
        max_steps=3,
        timeout_seconds=None,
        record=True,
    )
    assert isinstance(result.recorder, TrajectoryRecorder)
    assert len(result.recorder) == result.steps


# ---------------------------------------------------------------------------
# quick_safety_monitor
# ---------------------------------------------------------------------------


def test_quick_safety_monitor_returns_monitor() -> None:
    from agent_sim_bridge import quick_safety_monitor
    from agent_sim_bridge.safety.monitor import SafetyMonitor

    monitor = quick_safety_monitor([("joint_0", 0, -1.0, 1.0)])
    assert isinstance(monitor, SafetyMonitor)


def test_quick_safety_monitor_no_violations_within_bounds() -> None:
    from agent_sim_bridge import quick_safety_monitor

    monitor = quick_safety_monitor([("joint_0", 0, -1.0, 1.0)])
    monitor.start_monitoring()
    violations = monitor.check_step([0.5])
    assert violations == []


def test_quick_safety_monitor_violation_outside_bounds() -> None:
    from agent_sim_bridge import quick_safety_monitor

    monitor = quick_safety_monitor([("joint_0", 0, -1.0, 1.0)])
    monitor.start_monitoring()
    violations = monitor.check_step([2.0])
    assert len(violations) == 1
    assert violations[0].constraint_name == "joint_0"


def test_quick_safety_monitor_multiple_constraints() -> None:
    from agent_sim_bridge import quick_safety_monitor

    monitor = quick_safety_monitor([
        ("joint_0", 0, -1.0, 1.0),
        ("joint_1", 1, -1.0, 1.0),
    ])
    monitor.start_monitoring()
    # Both dimensions out of range
    violations = monitor.check_step([2.0, -2.0])
    assert len(violations) == 2


def test_quick_safety_monitor_emergency_stop_not_triggered_on_error() -> None:
    from agent_sim_bridge import quick_safety_monitor

    # ERROR severity does not trigger emergency stop (only CRITICAL does)
    monitor = quick_safety_monitor([("joint_0", 0, -1.0, 1.0)])
    monitor.start_monitoring()
    monitor.check_step([5.0])
    assert not monitor.emergency_stopped


def test_quick_safety_monitor_summary_structure() -> None:
    from agent_sim_bridge import quick_safety_monitor

    monitor = quick_safety_monitor([("joint_0", 0, -1.0, 1.0)])
    monitor.start_monitoring()
    monitor.check_step([0.5])
    summary = monitor.summary()
    assert "total_steps" in summary
    assert "total_violations" in summary
    assert "emergency_stopped" in summary


# ---------------------------------------------------------------------------
# quick_skill_library
# ---------------------------------------------------------------------------


def test_quick_skill_library_returns_library() -> None:
    from agent_sim_bridge import quick_skill_library
    from agent_sim_bridge.skills.library import SkillLibrary

    library = quick_skill_library()
    assert isinstance(library, SkillLibrary)


def test_quick_skill_library_empty_on_creation() -> None:
    from agent_sim_bridge import quick_skill_library

    library = quick_skill_library()
    assert len(library) == 0
    assert library.list_names() == []


def test_quick_skill_library_independent_instances() -> None:
    from agent_sim_bridge import quick_skill_library

    lib1 = quick_skill_library()
    lib2 = quick_skill_library()
    assert lib1 is not lib2


def test_quick_skill_library_repr() -> None:
    from agent_sim_bridge import quick_skill_library

    library = quick_skill_library()
    text = repr(library)
    assert "SkillLibrary" in text


# ---------------------------------------------------------------------------
# quick_gap_analysis
# ---------------------------------------------------------------------------


def test_quick_gap_analysis_returns_gap_report() -> None:
    from agent_sim_bridge import quick_gap_analysis
    from agent_sim_bridge.metrics.gap import GapReport

    sim_obs = [[1.0, 0.0], [1.1, 0.1]]
    real_obs = [[1.05, 0.02], [1.15, 0.12]]
    report = quick_gap_analysis(sim_obs, real_obs)
    assert isinstance(report, GapReport)


def test_quick_gap_analysis_non_negative_mae() -> None:
    from agent_sim_bridge import quick_gap_analysis

    sim_obs = [[1.0, 0.0], [1.1, 0.1]]
    real_obs = [[1.05, 0.02], [1.15, 0.12]]
    report = quick_gap_analysis(sim_obs, real_obs)
    assert report.observation_mae >= 0.0
    assert report.observation_rmse >= 0.0


def test_quick_gap_analysis_zero_gap_identical_trajectories() -> None:
    from agent_sim_bridge import quick_gap_analysis

    obs = [[1.0, 2.0], [3.0, 4.0]]
    report = quick_gap_analysis(obs, obs)
    assert report.observation_mae == pytest.approx(0.0, abs=1e-9)
    assert report.overall_gap_score == pytest.approx(0.0, abs=1e-9)


def test_quick_gap_analysis_with_rewards() -> None:
    from agent_sim_bridge import quick_gap_analysis

    sim_obs = [[1.0], [2.0]]
    real_obs = [[1.1], [2.1]]
    report = quick_gap_analysis(
        sim_obs, real_obs,
        sim_rewards=[1.0, 1.0],
        real_rewards=[0.9, 0.9],
    )
    assert report.reward_mae > 0.0


def test_quick_gap_analysis_summary_keys() -> None:
    from agent_sim_bridge import quick_gap_analysis

    report = quick_gap_analysis([[1.0]], [[1.0]])
    summary = report.summary()
    for key in ("observation_mae", "observation_rmse", "overall_gap_score"):
        assert key in summary


def test_quick_gap_analysis_empty_trajectories() -> None:
    from agent_sim_bridge import quick_gap_analysis

    # Empty input should not raise — returns a zero-gap report
    report = quick_gap_analysis([], [])
    assert report.n_sim_steps == 0
    assert report.n_real_steps == 0


def test_quick_gap_analysis_metadata_embedded() -> None:
    from agent_sim_bridge import quick_gap_analysis

    report = quick_gap_analysis([[1.0]], [[1.0]], metadata={"policy": "ppo-v1"})
    assert report.metadata.get("policy") == "ppo-v1"


# ---------------------------------------------------------------------------
# quick_transfer_bridge
# ---------------------------------------------------------------------------


def test_quick_transfer_bridge_returns_bridge() -> None:
    from agent_sim_bridge import quick_transfer_bridge
    from agent_sim_bridge.transfer.bridge import TransferBridge

    bridge = quick_transfer_bridge([1.0, 1.0])
    assert isinstance(bridge, TransferBridge)


def test_quick_transfer_bridge_identity_transform() -> None:
    from agent_sim_bridge import quick_transfer_bridge

    bridge = quick_transfer_bridge([1.0, 1.0], offsets=[0.0, 0.0])
    sim_values = [3.0, 4.0]
    real_values = bridge.sim_to_real(sim_values)
    assert real_values == pytest.approx(sim_values)


def test_quick_transfer_bridge_round_trip() -> None:
    from agent_sim_bridge import quick_transfer_bridge

    bridge = quick_transfer_bridge([1.05, 0.98], offsets=[0.01, -0.02])
    sim_values = [1.0, 2.0]
    real_values = bridge.sim_to_real(sim_values)
    recovered = bridge.real_to_sim(real_values)
    assert recovered == pytest.approx(sim_values, abs=1e-6)


def test_quick_transfer_bridge_default_zero_offsets() -> None:
    from agent_sim_bridge import quick_transfer_bridge

    bridge = quick_transfer_bridge([2.0, 3.0])
    real_values = bridge.sim_to_real([1.0, 1.0])
    assert real_values == pytest.approx([2.0, 3.0])


def test_quick_transfer_bridge_repr() -> None:
    from agent_sim_bridge import quick_transfer_bridge

    bridge = quick_transfer_bridge([1.0])
    text = repr(bridge)
    assert "TransferBridge" in text


def test_quick_transfer_bridge_zero_scale_raises() -> None:
    from agent_sim_bridge import quick_transfer_bridge

    with pytest.raises(ValueError):
        quick_transfer_bridge([0.0, 1.0])


# ---------------------------------------------------------------------------
# Top-level __init__ exports
# ---------------------------------------------------------------------------


def test_all_convenience_symbols_exported() -> None:
    import agent_sim_bridge as asb

    expected_symbols = [
        "Simulator",
        "quick_gap_analysis",
        "quick_recorder",
        "quick_safety_monitor",
        "quick_sandbox",
        "quick_skill_library",
        "quick_transfer_bridge",
    ]
    for symbol in expected_symbols:
        assert hasattr(asb, symbol), f"Missing top-level export: {symbol}"
        assert symbol in asb.__all__, f"Symbol not in __all__: {symbol}"


def test_quick_functions_importable_from_package() -> None:
    from agent_sim_bridge import (
        quick_gap_analysis,
        quick_recorder,
        quick_safety_monitor,
        quick_sandbox,
        quick_skill_library,
        quick_transfer_bridge,
    )

    assert callable(quick_gap_analysis)
    assert callable(quick_recorder)
    assert callable(quick_safety_monitor)
    assert callable(quick_sandbox)
    assert callable(quick_skill_library)
    assert callable(quick_transfer_bridge)
