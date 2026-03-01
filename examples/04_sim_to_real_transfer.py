#!/usr/bin/env python3
"""Example: Sim-to-Real Transfer and Domain Randomization

Demonstrates calibrating a simulation to match real-world data
and applying domain randomization to improve transfer.

Usage:
    python examples/04_sim_to_real_transfer.py

Requirements:
    pip install agent-sim-bridge
"""
from __future__ import annotations

import agent_sim_bridge
from agent_sim_bridge import (
    CalibrationProfile,
    Calibrator,
    DistributionType,
    DomainRandomizer,
    RandomizationConfig,
    TransferBridge,
)


def main() -> None:
    print(f"agent-sim-bridge version: {agent_sim_bridge.__version__}")

    # Step 1: Calibrate simulation against real-world observations
    calibrator = Calibrator()
    sim_observations = [
        {"friction": 0.80, "mass": 1.0, "response_time_ms": 120},
        {"friction": 0.82, "mass": 1.0, "response_time_ms": 118},
        {"friction": 0.79, "mass": 1.0, "response_time_ms": 125},
    ]
    real_observations = [
        {"friction": 0.65, "mass": 1.05, "response_time_ms": 145},
        {"friction": 0.67, "mass": 1.04, "response_time_ms": 150},
        {"friction": 0.64, "mass": 1.06, "response_time_ms": 148},
    ]
    profile: CalibrationProfile = calibrator.calibrate(
        sim_data=sim_observations,
        real_data=real_observations,
    )
    print(f"Calibration profile:")
    for param, factor in profile.scaling_factors.items():
        print(f"  {param}: scale={factor:.3f}")

    # Step 2: Apply transfer bridge
    bridge = TransferBridge(profile=profile)
    sim_obs = {"friction": 0.80, "mass": 1.0, "response_time_ms": 120}
    real_obs = bridge.transfer(sim_obs)
    print(f"\nSim observation: {sim_obs}")
    print(f"Transferred:     {real_obs}")

    # Step 3: Domain randomization
    randomization_config = RandomizationConfig(
        parameters={
            "friction": {
                "distribution": DistributionType.UNIFORM,
                "low": 0.50,
                "high": 0.80,
            },
            "mass": {
                "distribution": DistributionType.GAUSSIAN,
                "mean": 1.0,
                "std": 0.05,
            },
        }
    )
    randomizer = DomainRandomizer(config=randomization_config)
    print(f"\nDomain randomization (3 samples):")
    for i in range(3):
        sample = randomizer.sample()
        print(f"  Sample {i + 1}: friction={sample['friction']:.3f}, "
              f"mass={sample['mass']:.3f}")

    # Step 4: Gap analysis
    from agent_sim_bridge import SimRealGap, GapReport
    gap = SimRealGap()
    for sim_o, real_o in zip(sim_observations, real_observations):
        gap.record(sim_obs=sim_o, real_obs=real_o)
    report: GapReport = gap.report()
    print(f"\nSim-to-real gap report:")
    for param, gap_val in report.parameter_gaps.items():
        print(f"  {param}: gap={gap_val:.4f}")


if __name__ == "__main__":
    main()
