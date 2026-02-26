"""Sensor ABC and SensorReading dataclass.

All sensor implementations subclass :class:`Sensor` and produce
:class:`SensorReading` objects.  Readings carry the raw data, a confidence
score, and a timestamp so fusion algorithms can weight and order them.

SensorType enum
---------------
Enumerates the physical measurement categories supported by the bridge.
Backends and fusion logic use these labels to group compatible readings.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class SensorType(str, Enum):
    """Supported physical measurement categories."""

    POSITION = "position"
    VELOCITY = "velocity"
    FORCE = "force"
    TEMPERATURE = "temperature"
    VISION = "vision"
    LIDAR = "lidar"
    IMU = "imu"


@dataclass
class SensorReading:
    """A single reading produced by a :class:`Sensor`.

    Attributes
    ----------
    sensor_id:
        Unique identifier of the sensor that produced this reading.
    sensor_type:
        Physical measurement category.
    values:
        The measured data as a list of floats.  For scalar sensors this has
        length 1; for vector sensors (e.g., IMU) it may be longer.
    confidence:
        Reliability score in ``[0.0, 1.0]``.  Higher is more reliable.
        Fusion algorithms use this to weight readings.
    timestamp:
        Wall-clock time of the reading (seconds since epoch).
    metadata:
        Arbitrary sensor-specific diagnostic information.
    """

    sensor_id: str
    sensor_type: SensorType
    values: list[float]
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"SensorReading confidence must be in [0, 1], got {self.confidence}."
            )
        if not self.values:
            raise ValueError("SensorReading.values must not be empty.")

    @property
    def scalar(self) -> float:
        """Convenience accessor for single-value readings."""
        if len(self.values) != 1:
            raise ValueError(
                f"Cannot call .scalar on a multi-value reading (len={len(self.values)})."
            )
        return self.values[0]


class Sensor(ABC):
    """Abstract base class for all sensor implementations.

    Subclasses must implement :meth:`read` to return a :class:`SensorReading`.
    They may also override :meth:`calibrate` if the sensor supports
    in-place calibration.

    Parameters
    ----------
    sensor_id:
        Unique identifier for this sensor instance.
    sensor_type:
        Physical measurement category this sensor belongs to.
    """

    def __init__(self, sensor_id: str, sensor_type: SensorType) -> None:
        self._sensor_id = sensor_id
        self._sensor_type = sensor_type

    @property
    def sensor_id(self) -> str:
        """Unique sensor identifier."""
        return self._sensor_id

    @property
    def sensor_type(self) -> SensorType:
        """Measurement category."""
        return self._sensor_type

    @abstractmethod
    def read(self) -> SensorReading:
        """Acquire one reading from this sensor.

        Returns
        -------
        SensorReading
            The acquired measurement.
        """

    def calibrate(self) -> None:
        """Optional: perform in-place sensor calibration.

        The default implementation is a no-op.  Subclasses that support
        calibration should override this method.
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sensor_id={self._sensor_id!r}, "
            f"sensor_type={self._sensor_type.value!r})"
        )
