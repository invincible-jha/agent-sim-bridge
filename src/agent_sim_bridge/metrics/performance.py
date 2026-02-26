"""PerformanceTracker — lightweight in-memory performance metrics accumulation.

Records scalar metric values over time and provides statistical summaries
without external dependencies.  Suitable for tracking reward, step counts,
calibration errors, or any other numeric signal produced during training or
evaluation.

Design
------
Metrics are keyed by string name.  Each metric maintains a list of
(timestamp, value) pairs.  The summary computes min, max, mean, and
standard deviation using only the Python stdlib.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """One recorded value for a named metric.

    Attributes
    ----------
    name:
        Metric name.
    value:
        Recorded scalar value.
    timestamp:
        Wall-clock time of the recording (seconds since epoch).
    step:
        Optional step index (episode step, training iteration, etc.).
    tags:
        Arbitrary string labels for filtering.
    """

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    step: int | None = None
    tags: list[str] = field(default_factory=list)


def _statistics(values: list[float]) -> dict[str, float]:
    """Compute min, max, mean, std, and count for a list of floats."""
    n = len(values)
    if n == 0:
        return {"count": 0.0, "min": float("nan"), "max": float("nan"),
                "mean": float("nan"), "std": float("nan")}
    total = sum(values)
    mean = total / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "count": float(n),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(variance),
    }


class PerformanceTracker:
    """Accumulate and summarise performance metrics across episodes or steps.

    Parameters
    ----------
    max_records_per_metric:
        If set, each metric keeps at most this many records (oldest dropped
        first).  Pass ``None`` for unlimited history.

    Example
    -------
    ::

        tracker = PerformanceTracker()
        tracker.record("episode_reward", 42.3, step=1)
        tracker.record("episode_reward", 51.0, step=2)
        print(tracker.summary())
    """

    def __init__(self, max_records_per_metric: int | None = None) -> None:
        self._max_records = max_records_per_metric
        self._records: dict[str, list[MetricRecord]] = {}

    def record(
        self,
        name: str,
        value: float,
        step: int | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Append one value for metric ``name``.

        Parameters
        ----------
        name:
            Metric identifier.
        value:
            The scalar value to record.
        step:
            Optional step index.
        tags:
            Optional labels.
        """
        metric_record = MetricRecord(
            name=name,
            value=value,
            step=step,
            tags=tags or [],
        )
        if name not in self._records:
            self._records[name] = []
        bucket = self._records[name]
        bucket.append(metric_record)
        if self._max_records is not None and len(bucket) > self._max_records:
            bucket.pop(0)
        logger.debug("Recorded metric %r = %.6f (step=%s).", name, value, step)

    def record_many(self, values: dict[str, float], step: int | None = None) -> None:
        """Convenience method to record multiple metrics at once.

        Parameters
        ----------
        values:
            Mapping of metric name to value.
        step:
            Shared step index applied to all metrics.
        """
        for name, value in values.items():
            self.record(name, value, step=step)

    def get_values(self, name: str) -> list[float]:
        """Return all recorded values for metric ``name``.

        Parameters
        ----------
        name:
            Metric name.

        Returns
        -------
        list[float]
            Values in recording order.  Empty list if metric not yet recorded.
        """
        return [r.value for r in self._records.get(name, [])]

    def get_records(self, name: str) -> list[MetricRecord]:
        """Return all :class:`MetricRecord` objects for metric ``name``."""
        return list(self._records.get(name, []))

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute statistics for every recorded metric.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping of metric name to ``{count, min, max, mean, std}``.
        """
        return {
            name: _statistics(self.get_values(name))
            for name in sorted(self._records)
        }

    def metric_summary(self, name: str) -> dict[str, float]:
        """Compute statistics for one named metric.

        Parameters
        ----------
        name:
            Metric name.

        Returns
        -------
        dict[str, float]
            ``{count, min, max, mean, std}``.
        """
        return _statistics(self.get_values(name))

    def metric_names(self) -> list[str]:
        """Return sorted list of all metric names recorded so far."""
        return sorted(self._records)

    def reset(self, name: str | None = None) -> None:
        """Clear recorded data.

        Parameters
        ----------
        name:
            If given, clear only that metric.  Otherwise clear all metrics.
        """
        if name is not None:
            self._records.pop(name, None)
        else:
            self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"PerformanceTracker("
            f"n_metrics={len(self._records)}, "
            f"metrics={self.metric_names()})"
        )
