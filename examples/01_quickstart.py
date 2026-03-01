#!/usr/bin/env python3
"""Example: Quickstart — agent-sim-bridge

Minimal working example: run a simulation environment step,
record a trajectory, and check safety constraints.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-sim-bridge
"""
from __future__ import annotations

import agent_sim_bridge
from agent_sim_bridge import (
    Simulator,
    SimulationEnvironment,
    SafetyConstraint,
    SafetyChecker,
    ConstraintType,
    ViolationSeverity,
    TrajectoryRecorder,
    TrajectoryStep,
)


def main() -> None:
    print(f"agent-sim-bridge version: {agent_sim_bridge.__version__}")

    # Step 1: Use the convenience Simulator
    sim = Simulator()
    print(f"Simulator ready: {sim}")

    # Step 2: Run a simulation environment
    env = SimulationEnvironment(
        name="warehouse-sim",
        max_steps=5,
    )
    obs = env.reset()
    print(f"\nEnvironment '{env.name}' reset. Observation: {obs}")

    for step_num in range(3):
        action = {"move": "forward", "speed": 0.5}
        result = env.step(action=action)
        print(f"  Step {step_num + 1}: done={result.done}, "
              f"reward={result.reward:.2f}")
        if result.done:
            break

    # Step 3: Record a trajectory
    recorder = TrajectoryRecorder(episode_id="ep-001")
    for i in range(3):
        recorder.record(TrajectoryStep(
            step=i,
            observation={"position": [float(i), 0.0, 0.0]},
            action={"move": "forward"},
            reward=1.0,
        ))
    trajectory = recorder.finish()
    print(f"\nTrajectory: {trajectory.episode_id}, "
          f"steps={len(trajectory.steps)}")

    # Step 4: Safety constraint check
    constraint = SafetyConstraint(
        name="max-speed",
        constraint_type=ConstraintType.UPPER_BOUND,
        parameter="speed",
        threshold=1.0,
        severity=ViolationSeverity.CRITICAL,
    )
    checker = SafetyChecker(constraints=[constraint])
    violation = checker.check({"speed": 1.5})
    print(f"\nSafety check (speed=1.5): violated={violation is not None}")
    if violation:
        print(f"  Violation: [{violation.severity.value}] {violation.name}")


if __name__ == "__main__":
    main()
