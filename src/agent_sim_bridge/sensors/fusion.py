"""SensorFusion — combine multiple sensor readings into one estimate.

Supported fusion strategies are defined in the :class:`FusionStrategy` enum.
The default is confidence-weighted averaging, which is robust and requires no
training data.

All fusion methods operate on :class:`~agent_sim_bridge.sensors.base.SensorReading`
objects and return a new reading whose ``sensor_id`` is ``"fused"`` and whose
confidence is derived from the input confidences.
"""
from __future__ import annotations

import logging
import time
from enum import Enum

from agent_sim_bridge.sensors.base import SensorReading, SensorType

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Supported sensor fusion algorithms."""

    WEIGHTED_AVERAGE = "weighted_average"
    SIMPLE_AVERAGE = "simple_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    LOWEST_CONFIDENCE = "lowest_confidence"


class SensorFusion:
    """Fuse a list of :class:`~agent_sim_bridge.sensors.base.SensorReading` objects.

    Parameters
    ----------
    strategy:
        Fusion algorithm to use.  Defaults to :attr:`FusionStrategy.WEIGHTED_AVERAGE`.
    output_sensor_type:
        The :class:`~agent_sim_bridge.sensors.base.SensorType` to assign to the
        fused reading.  If ``None``, the type is inferred from the first input.

    Example
    -------
    ::

        fusion = SensorFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
        fused = fusion.fuse([reading_a, reading_b, reading_c])
    """

    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        output_sensor_type: SensorType | None = None,
    ) -> None:
        self._strategy = strategy
        self._output_sensor_type = output_sensor_type

    @property
    def strategy(self) -> FusionStrategy:
        """Active fusion strategy."""
        return self._strategy

    def fuse(self, readings: list[SensorReading]) -> SensorReading:
        """Combine ``readings`` into a single fused reading.

        Parameters
        ----------
        readings:
            List of readings to fuse.  Must be non-empty and all readings
            must have the same number of values.

        Returns
        -------
        SensorReading
            Fused reading with ``sensor_id="fused"``.

        Raises
        ------
        ValueError
            If ``readings`` is empty or dimensionalities are inconsistent.
        """
        if not readings:
            raise ValueError("Cannot fuse an empty list of readings.")

        n_values = len(readings[0].values)
        for index, reading in enumerate(readings):
            if len(reading.values) != n_values:
                raise ValueError(
                    f"readings[{index}] has {len(reading.values)} values; "
                    f"expected {n_values} (same as readings[0])."
                )

        sensor_type = self._output_sensor_type or readings[0].sensor_type

        if self._strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average(readings, sensor_type)
        if self._strategy == FusionStrategy.SIMPLE_AVERAGE:
            return self._simple_average(readings, sensor_type)
        if self._strategy == FusionStrategy.HIGHEST_CONFIDENCE:
            return self._pick_by_confidence(readings, sensor_type, pick_highest=True)
        if self._strategy == FusionStrategy.LOWEST_CONFIDENCE:
            return self._pick_by_confidence(readings, sensor_type, pick_highest=False)

        raise ValueError(f"Unknown fusion strategy: {self._strategy!r}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_average(
        readings: list[SensorReading],
        sensor_type: SensorType,
    ) -> SensorReading:
        """Confidence-weighted average of all readings."""
        total_weight = sum(r.confidence for r in readings)
        if total_weight == 0.0:
            # Fall back to equal weighting when all confidences are zero.
            weights = [1.0 / len(readings)] * len(readings)
        else:
            weights = [r.confidence / total_weight for r in readings]

        n_values = len(readings[0].values)
        fused_values = [
            sum(weights[i] * readings[i].values[dim] for i in range(len(readings)))
            for dim in range(n_values)
        ]
        avg_confidence = sum(r.confidence for r in readings) / len(readings)

        return SensorReading(
            sensor_id="fused",
            sensor_type=sensor_type,
            values=fused_values,
            confidence=avg_confidence,
            timestamp=time.time(),
            metadata={"strategy": FusionStrategy.WEIGHTED_AVERAGE.value, "n_fused": len(readings)},
        )

    @staticmethod
    def _simple_average(
        readings: list[SensorReading],
        sensor_type: SensorType,
    ) -> SensorReading:
        """Unweighted average of all readings."""
        n = len(readings)
        n_values = len(readings[0].values)
        fused_values = [
            sum(readings[i].values[dim] for i in range(n)) / n
            for dim in range(n_values)
        ]
        avg_confidence = sum(r.confidence for r in readings) / n

        return SensorReading(
            sensor_id="fused",
            sensor_type=sensor_type,
            values=fused_values,
            confidence=avg_confidence,
            timestamp=time.time(),
            metadata={"strategy": FusionStrategy.SIMPLE_AVERAGE.value, "n_fused": n},
        )

    @staticmethod
    def _pick_by_confidence(
        readings: list[SensorReading],
        sensor_type: SensorType,
        pick_highest: bool,
    ) -> SensorReading:
        """Return the reading with the highest (or lowest) confidence."""
        chosen = max(readings, key=lambda r: r.confidence) if pick_highest else min(
            readings, key=lambda r: r.confidence
        )
        strategy_name = (
            FusionStrategy.HIGHEST_CONFIDENCE.value
            if pick_highest
            else FusionStrategy.LOWEST_CONFIDENCE.value
        )
        return SensorReading(
            sensor_id="fused",
            sensor_type=sensor_type,
            values=list(chosen.values),
            confidence=chosen.confidence,
            timestamp=time.time(),
            metadata={
                "strategy": strategy_name,
                "source_sensor_id": chosen.sensor_id,
            },
        )

    def __repr__(self) -> str:
        return f"SensorFusion(strategy={self._strategy.value!r})"
