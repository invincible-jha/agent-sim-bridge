"""Sensor abstractions — base classes, fusion, and noise models.

This package provides the building blocks for modelling real and simulated
sensor pipelines:

* **base** — :class:`Sensor` ABC, :class:`SensorReading` dataclass,
  :class:`SensorType` enum.
* **fusion** — :class:`SensorFusion` with multiple strategies.
* **noise** — :class:`GaussianNoise`, :class:`UniformNoise`,
  :class:`CompositeNoise`.
"""
from __future__ import annotations

from agent_sim_bridge.sensors.base import Sensor, SensorReading, SensorType
from agent_sim_bridge.sensors.fusion import FusionStrategy, SensorFusion
from agent_sim_bridge.sensors.noise import CompositeNoise, GaussianNoise, NoiseModel, UniformNoise

__all__ = [
    "Sensor",
    "SensorReading",
    "SensorType",
    "SensorFusion",
    "FusionStrategy",
    "NoiseModel",
    "GaussianNoise",
    "UniformNoise",
    "CompositeNoise",
]
