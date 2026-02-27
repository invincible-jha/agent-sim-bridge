"""Convenience API for agent-sim-bridge — 3-line quickstart.

Example
-------
::

    # Option 1 — zero-config Simulator class
    from agent_sim_bridge import Simulator
    sim = Simulator()
    result = sim.run(steps=10)
    print(result.total_reward)

    # Option 2 — functional quick-* helpers
    from agent_sim_bridge import quick_recorder, quick_safety_monitor
    from agent_sim_bridge import quick_skill_library, quick_gap_analysis
    from agent_sim_bridge import quick_transfer_bridge

    recorder = quick_recorder()
    recorder.record(obs, action, 1.0, next_obs, False, False)
    recorder.save("episode.npz")

    monitor = quick_safety_monitor([("joint_0", 0, -1.0, 1.0)])
    monitor.start_monitoring()
    violations = monitor.check_step([0.5])

    report = quick_gap_analysis(sim_obs, real_obs)
    print(report.overall_gap_score)

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from agent_sim_bridge.environment.base import SpaceSpec


# ---------------------------------------------------------------------------
# Minimal stub backend — satisfies the SimBackend protocol without any
# external physics engine dependency.
# ---------------------------------------------------------------------------


class _StubBackend:
    """Zero-dependency stub that satisfies the SimBackend protocol.

    Returns random observations and a constant unit reward per step so
    callers can exercise the full sandbox machinery without wiring up a
    real physics engine.
    """

    _OBS_DIM: int = 4
    _ACT_DIM: int = 2

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._state_space = SpaceSpec(shape=(self._OBS_DIM,))
        self._action_space = SpaceSpec(shape=(self._ACT_DIM,))

    # -- SimBackend protocol -----------------------------------------------

    def backend_reset(
        self,
        seed: int | None,
        options: dict[str, object] | None,
    ) -> NDArray[np.float32]:
        """Reset the stub RNG and return a random initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self._rng.random(self._OBS_DIM).astype(np.float32)

    def backend_step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        """Advance one step: return random obs, unit reward, no termination."""
        obs = self._rng.random(self._OBS_DIM).astype(np.float32)
        reward: float = 1.0
        terminated: bool = False
        truncated: bool = False
        info: dict[str, object] = {}
        return obs, reward, terminated, truncated, info

    def backend_observe(self) -> NDArray[np.float32]:
        """Return a random observation without advancing the clock."""
        return self._rng.random(self._OBS_DIM).astype(np.float32)

    def backend_act(self, action: NDArray[np.float32]) -> None:
        """Accept an action without computing a reward (no-op for stub)."""

    def backend_close(self) -> None:
        """Release resources (no-op for stub)."""

    @property
    def backend_state_space(self) -> SpaceSpec:
        """Observation space spec."""
        return self._state_space

    @property
    def backend_action_space(self) -> SpaceSpec:
        """Action space spec."""
        return self._action_space


# ---------------------------------------------------------------------------
# Simulator convenience class
# ---------------------------------------------------------------------------


class Simulator:
    """Zero-config simulator wrapper for the 80% use case.

    Bundles a stub (or user-supplied) environment with
    :class:`~agent_sim_bridge.simulation.sandbox.SimulationSandbox`
    and optionally a :class:`~agent_sim_bridge.simulation.scenario.Scenario`
    to provide a one-call ``run()`` interface.

    Parameters
    ----------
    scenario:
        Optional :class:`~agent_sim_bridge.simulation.scenario.Scenario`
        instance, or a plain scenario name string.  When ``None`` a default
        ``"quickstart-scenario"`` is used (100 steps, no outcome criteria).
    seed:
        RNG seed forwarded to the stub environment for reproducibility.

    Example
    -------
    ::

        from agent_sim_bridge import Simulator

        sim = Simulator(scenario="navigation-basic")
        result = sim.run(steps=50)
        print(result.total_reward, result.steps)
        print(sim.outcome_met(result))
    """

    def __init__(
        self,
        scenario: "Scenario | str | None" = None,
        seed: int | None = None,
    ) -> None:
        from agent_sim_bridge.simulation.scenario import Scenario

        if scenario is None:
            self._scenario: Scenario = Scenario(name="quickstart-scenario")
        elif isinstance(scenario, str):
            self._scenario = Scenario(name=scenario)
        else:
            self._scenario = scenario

        self._seed = seed
        self._backend = _StubBackend(seed=seed)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, steps: int = 50) -> "ExecutionResult":
        """Run a simulation episode and return the result.

        A zero-action policy is used so no ML model is required.

        Parameters
        ----------
        steps:
            Maximum number of timesteps to execute before truncation.

        Returns
        -------
        ExecutionResult
            Episode statistics: ``total_reward``, ``steps``, ``success``,
            ``terminated``, ``truncated``, ``wall_time_seconds``.
        """
        from agent_sim_bridge.environment.sim_env import SimulationEnvironment
        from agent_sim_bridge.simulation.sandbox import SimulationSandbox

        env = SimulationEnvironment(
            backend=self._backend,
            name="quickstart-env",
            max_episode_steps=steps,
        )

        sandbox = SimulationSandbox(
            environment=env,
            max_steps=steps,
            timeout_seconds=None,
            reset_seed=self._seed,
        )

        act_shape = self._backend.backend_action_space.shape

        def _zero_policy(obs: NDArray[np.float32]) -> NDArray[np.float32]:
            return np.zeros(act_shape, dtype=np.float32)

        return sandbox.run(policy=_zero_policy)

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> "Scenario":
        """The active :class:`~agent_sim_bridge.simulation.scenario.Scenario`."""
        return self._scenario

    def outcome_met(self, result: "ExecutionResult") -> bool:
        """Check whether *result* satisfies the scenario's outcome criteria.

        Parameters
        ----------
        result:
            :class:`~agent_sim_bridge.simulation.sandbox.ExecutionResult`
            returned by :meth:`run`.

        Returns
        -------
        bool
            True if all outcome criteria defined on the scenario are met.
        """
        return self._scenario.evaluate(
            total_reward=result.total_reward,
            steps=result.steps,
            terminated=result.terminated,
        )

    def __repr__(self) -> str:
        return f"Simulator(scenario={self._scenario.name!r}, seed={self._seed!r})"


# ---------------------------------------------------------------------------
# quick_* functional helpers
# ---------------------------------------------------------------------------


def quick_recorder(max_steps: int | None = None) -> "TrajectoryRecorder":
    """Create a ready-to-use trajectory recorder.

    Returns a :class:`~agent_sim_bridge.simulation.recorder.TrajectoryRecorder`
    with an optional step cap.  No configuration required — call
    :meth:`~agent_sim_bridge.simulation.recorder.TrajectoryRecorder.record`
    immediately after each environment step.

    Parameters
    ----------
    max_steps:
        Maximum number of steps to record.  ``None`` means no limit.

    Returns
    -------
    TrajectoryRecorder
        An empty recorder ready to accumulate steps.

    Example
    -------
    ::

        recorder = quick_recorder()
        recorder.record(obs, action, 1.0, next_obs, False, False)
        recorder.save("my_episode.npz")
        print(len(recorder))  # 1
    """
    from agent_sim_bridge.simulation.recorder import TrajectoryRecorder

    return TrajectoryRecorder(max_steps=max_steps)


def quick_sandbox(
    environment: "Environment",
    policy: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    max_steps: int = 1000,
    timeout_seconds: float | None = 60.0,
    record: bool = False,
    seed: int | None = None,
) -> "ExecutionResult":
    """Run a single sandboxed episode and return the result.

    Wraps :class:`~agent_sim_bridge.simulation.sandbox.SimulationSandbox`
    for the common case of running one episode with a policy callable.
    The sandbox enforces step and wall-clock limits automatically.

    Parameters
    ----------
    environment:
        Any :class:`~agent_sim_bridge.environment.base.Environment` instance.
    policy:
        Callable that maps an observation array to an action array.
    max_steps:
        Hard cap on episode length before truncation.
    timeout_seconds:
        Maximum wall-clock seconds for the episode.  Pass ``None`` to
        disable the time limit.
    record:
        When ``True`` the returned
        :class:`~agent_sim_bridge.simulation.sandbox.ExecutionResult`
        includes a populated
        :class:`~agent_sim_bridge.simulation.recorder.TrajectoryRecorder`.
    seed:
        RNG seed forwarded to the environment's ``reset()`` call for
        reproducible episodes.

    Returns
    -------
    ExecutionResult
        Episode statistics: total reward, step count, success flag, and
        optionally a trajectory recorder.

    Example
    -------
    ::

        result = quick_sandbox(env, policy=lambda obs: np.zeros(2, dtype="float32"))
        print(result.total_reward, result.steps, result.success)
    """
    from agent_sim_bridge.simulation.sandbox import SimulationSandbox

    sandbox = SimulationSandbox(
        environment=environment,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        record=record,
        reset_seed=seed,
    )
    return sandbox.run(policy=policy)


def quick_safety_monitor(
    constraints: list[tuple[str, int, float, float]],
    auto_stop_on_critical: bool = True,
) -> "SafetyMonitor":
    """Build a :class:`~agent_sim_bridge.safety.monitor.SafetyMonitor` from a compact spec.

    Each element of *constraints* is a 4-tuple
    ``(name, dimension, min_value, max_value)`` that becomes a
    :class:`~agent_sim_bridge.safety.constraints.SafetyConstraint` with
    ``ConstraintType.RANGE`` and ``ViolationSeverity.ERROR``.

    For advanced constraint types (MAX_RATE, FORBIDDEN_ZONE, CUSTOM) or
    custom severities, construct
    :class:`~agent_sim_bridge.safety.constraints.SafetyConstraint` objects
    directly and pass them to
    :class:`~agent_sim_bridge.safety.monitor.SafetyMonitor`.

    Parameters
    ----------
    constraints:
        List of ``(name, dimension, min_value, max_value)`` tuples.
    auto_stop_on_critical:
        Trigger emergency stop automatically on the first CRITICAL violation.

    Returns
    -------
    SafetyMonitor
        A configured monitor.  Call
        :meth:`~agent_sim_bridge.safety.monitor.SafetyMonitor.start_monitoring`
        before the episode loop, then
        :meth:`~agent_sim_bridge.safety.monitor.SafetyMonitor.check_step`
        on every action.

    Example
    -------
    ::

        monitor = quick_safety_monitor([
            ("joint_0", 0, -1.0, 1.0),
            ("joint_1", 1, -1.0, 1.0),
        ])
        monitor.start_monitoring()
        violations = monitor.check_step([0.5, 2.0])
        if monitor.emergency_stopped:
            print("Episode halted!")
    """
    from agent_sim_bridge.safety.constraints import (
        ConstraintType,
        SafetyConstraint,
        ViolationSeverity,
    )
    from agent_sim_bridge.safety.monitor import SafetyMonitor

    built: list[SafetyConstraint] = [
        SafetyConstraint(
            name=name,
            constraint_type=ConstraintType.RANGE,
            dimension=dimension,
            min_value=min_value,
            max_value=max_value,
            severity=ViolationSeverity.ERROR,
        )
        for name, dimension, min_value, max_value in constraints
    ]
    return SafetyMonitor(constraints=built, auto_stop_on_critical=auto_stop_on_critical)


def quick_skill_library() -> "SkillLibrary":
    """Create an empty :class:`~agent_sim_bridge.skills.library.SkillLibrary`.

    The library stores and retrieves
    :class:`~agent_sim_bridge.skills.base.Skill` instances by name.
    Register skills immediately and use
    :meth:`~agent_sim_bridge.skills.library.SkillLibrary.get` to retrieve
    them by exact name, or
    :meth:`~agent_sim_bridge.skills.library.SkillLibrary.search` for
    fuzzy lookup by name or tags.

    Returns
    -------
    SkillLibrary
        An empty skill registry.

    Example
    -------
    ::

        library = quick_skill_library()
        library.register(my_reach_skill)
        library.register(my_grasp_skill)
        skill = library.get("reach")
        print(library.list_names())
    """
    from agent_sim_bridge.skills.library import SkillLibrary

    return SkillLibrary()


def quick_gap_analysis(
    sim_observations: list[list[float]],
    real_observations: list[list[float]],
    sim_rewards: list[float] | None = None,
    real_rewards: list[float] | None = None,
    metadata: dict[str, object] | None = None,
) -> "GapReport":
    """Measure the sim-to-real observation and reward gap in one call.

    Wraps :class:`~agent_sim_bridge.metrics.gap.SimRealGap` for the common
    case of comparing two parallel trajectories collected under the same
    policy.

    Parameters
    ----------
    sim_observations:
        Observation vectors from the simulation trajectory.  Each element
        is a list of floats representing one timestep.
    real_observations:
        Observation vectors from the real trajectory.  Inner dimension must
        match ``sim_observations``.
    sim_rewards:
        Scalar rewards from simulation (optional).  When provided together
        with ``real_rewards``, reward MAE and bias are included in the report.
    real_rewards:
        Scalar rewards from the real system (optional).
    metadata:
        Arbitrary annotations embedded in the returned
        :class:`~agent_sim_bridge.metrics.gap.GapReport` (e.g. policy name,
        date, environment version).

    Returns
    -------
    GapReport
        Computed gap metrics including ``observation_mae``,
        ``observation_rmse``, ``reward_mae``, ``reward_bias``,
        ``trajectory_length_ratio``, and ``overall_gap_score``.

    Example
    -------
    ::

        sim_obs = [[1.0, 0.0], [1.1, 0.1]]
        real_obs = [[1.05, 0.02], [1.15, 0.12]]
        report = quick_gap_analysis(sim_obs, real_obs)
        print(report.overall_gap_score)   # lower is better
        print(report.summary())
    """
    from agent_sim_bridge.metrics.gap import SimRealGap

    gap_metric = SimRealGap(metadata=metadata)
    return gap_metric.measure_gap(
        sim_observations=sim_observations,
        real_observations=real_observations,
        sim_rewards=sim_rewards,
        real_rewards=real_rewards,
    )


def quick_transfer_bridge(
    scale_factors: list[float],
    offsets: list[float] | None = None,
) -> "TransferBridge":
    """Create a sim-to-real :class:`~agent_sim_bridge.transfer.bridge.TransferBridge`.

    Constructs a :class:`~agent_sim_bridge.transfer.bridge.CalibrationProfile`
    with the given per-dimension linear transform and wraps it in a
    :class:`~agent_sim_bridge.transfer.bridge.TransferBridge` for immediate use.

    Parameters
    ----------
    scale_factors:
        Per-dimension multiplicative scale applied as
        ``real[i] = sim[i] * scale[i] + offset[i]``.
    offsets:
        Per-dimension additive offsets.  Defaults to all zeros when ``None``.

    Returns
    -------
    TransferBridge
        A configured bridge.  Use
        :meth:`~agent_sim_bridge.transfer.bridge.TransferBridge.sim_to_real`
        and
        :meth:`~agent_sim_bridge.transfer.bridge.TransferBridge.real_to_sim`
        to convert observation or action vectors.

    Raises
    ------
    ValueError
        If ``scale_factors`` is empty or contains a zero (which would make
        the inverse transform undefined).

    Example
    -------
    ::

        bridge = quick_transfer_bridge([1.05, 0.98], offsets=[0.01, -0.02])
        real_obs = bridge.sim_to_real([1.0, 2.0])
        sim_obs  = bridge.real_to_sim(real_obs)
        print(real_obs)   # [1.06, 1.94]
    """
    from agent_sim_bridge.transfer.bridge import CalibrationProfile, TransferBridge

    resolved_offsets: list[float] = (
        offsets if offsets is not None else [0.0] * len(scale_factors)
    )
    profile = CalibrationProfile(
        scale_factors=scale_factors,
        offsets=resolved_offsets,
    )
    return TransferBridge(profile=profile)


# ---------------------------------------------------------------------------
# TYPE_CHECKING imports — for type checkers and IDEs only.
# All runtime imports are deferred inside each function to avoid circular
# dependencies and keep the module lightweight.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from agent_sim_bridge.environment.base import Environment
    from agent_sim_bridge.metrics.gap import GapReport
    from agent_sim_bridge.safety.monitor import SafetyMonitor
    from agent_sim_bridge.simulation.recorder import TrajectoryRecorder
    from agent_sim_bridge.simulation.sandbox import ExecutionResult
    from agent_sim_bridge.simulation.scenario import Scenario
    from agent_sim_bridge.skills.library import SkillLibrary
    from agent_sim_bridge.transfer.bridge import TransferBridge


__all__: list[str] = [
    # class
    "Simulator",
    # functional helpers
    "quick_gap_analysis",
    "quick_recorder",
    "quick_safety_monitor",
    "quick_sandbox",
    "quick_skill_library",
    "quick_transfer_bridge",
]
