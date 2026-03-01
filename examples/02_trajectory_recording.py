#!/usr/bin/env python3
"""Example: Trajectory Recording and Replay

Demonstrates recording simulation trajectories, saving them,
and replaying episodes for analysis or training.

Usage:
    python examples/02_trajectory_recording.py

Requirements:
    pip install agent-sim-bridge
"""
from __future__ import annotations

import agent_sim_bridge
from agent_sim_bridge import (
    TrajectoryRecorder,
    TrajectoryReplay,
    TrajectoryStep,
)


def record_episode(episode_id: str, num_steps: int) -> object:
    """Record a synthetic episode trajectory."""
    recorder = TrajectoryRecorder(episode_id=episode_id)
    for i in range(num_steps):
        step = TrajectoryStep(
            step=i,
            observation={
                "position": [float(i) * 0.5, 0.0, 0.0],
                "velocity": [0.5, 0.0, 0.0],
            },
            action={"move": "forward", "speed": 0.5},
            reward=1.0 if i < num_steps - 1 else 10.0,
            done=(i == num_steps - 1),
            info={"step_cost": 0.01},
        )
        recorder.record(step)
    return recorder.finish()


def main() -> None:
    print(f"agent-sim-bridge version: {agent_sim_bridge.__version__}")

    # Record multiple episodes
    episodes = []
    for i in range(3):
        trajectory = record_episode(f"ep-{i:03d}", num_steps=5 + i)
        episodes.append(trajectory)
        total_reward = sum(s.reward for s in trajectory.steps)
        print(f"Episode {trajectory.episode_id}: "
              f"{len(trajectory.steps)} steps, "
              f"total_reward={total_reward:.1f}")

    # Replay the best episode (most steps = most reward here)
    best = max(episodes, key=lambda t: len(t.steps))
    print(f"\nReplaying best episode: {best.episode_id}")
    replay = TrajectoryReplay(trajectory=best)
    for step in replay.iter_steps():
        pos = step.observation.get("position", [])
        print(f"  Step {step.step}: position={pos}, "
              f"reward={step.reward:.1f}, done={step.done}")

    # Summary statistics
    print(f"\nReplay stats:")
    print(f"  Episode: {best.episode_id}")
    print(f"  Steps: {replay.total_steps()}")
    print(f"  Total reward: {replay.total_reward():.1f}")
    print(f"  Avg reward per step: {replay.avg_reward():.2f}")


if __name__ == "__main__":
    main()
