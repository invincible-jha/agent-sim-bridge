#!/usr/bin/env python3
"""Example: Sensor Fusion and Noise Modelling

Demonstrates sensor reading generation, Gaussian/uniform noise,
and fusion of multiple sensor readings.

Usage:
    python examples/05_sensor_fusion.py

Requirements:
    pip install agent-sim-bridge
"""
from __future__ import annotations

import agent_sim_bridge
from agent_sim_bridge import (
    CompositeNoise,
    FusionStrategy,
    GaussianNoise,
    Sensor,
    SensorFusion,
    SensorReading,
    SensorType,
    UniformNoise,
)


def make_sensor(sensor_id: str, sensor_type: SensorType) -> Sensor:
    """Create a sensor with Gaussian noise."""
    return Sensor(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        noise_model=GaussianNoise(mean=0.0, std=0.05),
    )


def main() -> None:
    print(f"agent-sim-bridge version: {agent_sim_bridge.__version__}")

    # Step 1: Create sensors with noise models
    lidar = make_sensor("lidar-front", SensorType.LIDAR)
    camera = make_sensor("camera-rgb", SensorType.CAMERA)
    imu = make_sensor("imu-main", SensorType.IMU)

    # Step 2: Generate noisy readings
    true_position = {"x": 5.0, "y": 3.0, "z": 0.0}
    print("Noisy sensor readings (true position x=5.0, y=3.0):")
    readings: list[SensorReading] = []
    for sensor in [lidar, camera, imu]:
        reading = sensor.read(ground_truth=true_position)
        readings.append(reading)
        print(f"  [{sensor.sensor_type.value}] "
              f"x={reading.data.get('x', 0):.3f}, "
              f"y={reading.data.get('y', 0):.3f}")

    # Step 3: Fuse readings
    for strategy in [FusionStrategy.AVERAGE, FusionStrategy.WEIGHTED]:
        fusion = SensorFusion(strategy=strategy)
        fused = fusion.fuse(readings=readings)
        print(f"\n  [{strategy.value}] fused: "
              f"x={fused.data.get('x', 0):.3f}, "
              f"y={fused.data.get('y', 0):.3f}")

    # Step 4: Composite noise
    composite = CompositeNoise(models=[
        GaussianNoise(mean=0.0, std=0.02),
        UniformNoise(low=-0.01, high=0.01),
    ])
    noisy = composite.apply(true_position)
    print(f"\nComposite noise applied:")
    print(f"  True: x=5.000, y=3.000")
    print(f"  Noisy: x={noisy.get('x', 0):.3f}, y={noisy.get('y', 0):.3f}")


if __name__ == "__main__":
    main()
