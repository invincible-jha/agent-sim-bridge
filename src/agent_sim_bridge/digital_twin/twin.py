"""Digital twin for agent configuration comparison.

Replays recorded production traffic against a modified agent configuration
and compares outputs to the original.  Produces a :class:`TwinReport`
describing match rate, divergence points, and performance delta.

This module uses no external inference engines — the "agent" is represented
as a callable so any backend can be plugged in.
"""
from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayRecord:
    """One item of production traffic to replay.

    Attributes
    ----------
    record_id:
        Unique identifier for this traffic record.
    input_data:
        The original input fed to the production agent.
    original_output:
        The output produced by the original production agent.
    original_latency_ms:
        Latency of the original production call in milliseconds.
    metadata:
        Optional additional context (e.g. user_id, session_id).
    """

    record_id: str
    input_data: object
    original_output: object
    original_latency_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DivergencePoint:
    """A single record where the modified agent diverged from original.

    Attributes
    ----------
    record_id:
        The :class:`ReplayRecord` identifier.
    original_output:
        What the original agent returned.
    modified_output:
        What the modified agent returned.
    latency_original_ms:
        Original call latency in milliseconds.
    latency_modified_ms:
        Modified call latency in milliseconds.
    """

    record_id: str
    original_output: object
    modified_output: object
    latency_original_ms: float
    latency_modified_ms: float


@dataclass(frozen=True)
class TwinReport:
    """Report comparing original vs modified agent behaviour.

    Attributes
    ----------
    total_records:
        Total number of replay records processed.
    matched_records:
        Number of records where outputs were identical.
    diverged_records:
        Number of records where outputs differed.
    match_rate:
        Fraction of matched records: matched / total, in [0.0, 1.0].
    divergence_points:
        Details of each diverging record.
    performance_delta_ms:
        Average latency of modified - average latency of original (ms).
        Negative means the modified version is faster.
    avg_latency_original_ms:
        Average latency of original calls across all records.
    avg_latency_modified_ms:
        Average latency of modified calls across all records.
    generated_at:
        UTC timestamp of report generation.
    """

    total_records: int
    matched_records: int
    diverged_records: int
    match_rate: float
    divergence_points: list[DivergencePoint]
    performance_delta_ms: float
    avg_latency_original_ms: float
    avg_latency_modified_ms: float
    generated_at: datetime.datetime


# ---------------------------------------------------------------------------
# Digital twin
# ---------------------------------------------------------------------------


class DigitalTwin:
    """Replay production traffic against a modified agent and compare outputs.

    The twin accepts two callables:
    - ``modified_agent``: the new or modified agent to evaluate.
    - ``output_comparator``: optional function to compare outputs (default:
      strict equality ``==``).

    Example
    -------
    ::

        def original_agent(inp):
            return {"answer": inp["question"].upper()}

        def modified_agent(inp):
            return {"answer": inp["question"].lower()}

        records = [
            ReplayRecord("r1", {"question": "Hello"}, {"answer": "HELLO"}),
            ReplayRecord("r2", {"question": "World"}, {"answer": "WORLD"}),
        ]
        twin = DigitalTwin(modified_agent=modified_agent)
        report = twin.replay(records)
        print(report.match_rate)   # 0.0 — all diverge
    """

    def __init__(
        self,
        modified_agent: Callable[[object], object],
        output_comparator: Callable[[object], bool] | None = None,
    ) -> None:
        self._modified_agent = modified_agent
        self._comparator = output_comparator or self._default_comparator

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def replay(self, records: list[ReplayRecord]) -> TwinReport:
        """Replay all *records* through the modified agent.

        Parameters
        ----------
        records:
            The production traffic records to replay.

        Returns
        -------
        TwinReport
            Comparison report.
        """
        if not records:
            return TwinReport(
                total_records=0,
                matched_records=0,
                diverged_records=0,
                match_rate=1.0,
                divergence_points=[],
                performance_delta_ms=0.0,
                avg_latency_original_ms=0.0,
                avg_latency_modified_ms=0.0,
                generated_at=datetime.datetime.now(datetime.timezone.utc),
            )

        matched = 0
        diverged = 0
        divergence_points: list[DivergencePoint] = []
        modified_latencies: list[float] = []
        original_latencies: list[float] = []

        for record in records:
            modified_output, modified_latency_ms = self._invoke_modified(record)
            original_latencies.append(record.original_latency_ms)
            modified_latencies.append(modified_latency_ms)

            if self._comparator(record.original_output, modified_output):
                matched += 1
            else:
                diverged += 1
                divergence_points.append(
                    DivergencePoint(
                        record_id=record.record_id,
                        original_output=record.original_output,
                        modified_output=modified_output,
                        latency_original_ms=record.original_latency_ms,
                        latency_modified_ms=modified_latency_ms,
                    )
                )

        total = len(records)
        match_rate = matched / total if total > 0 else 1.0
        avg_orig = sum(original_latencies) / len(original_latencies)
        avg_mod = sum(modified_latencies) / len(modified_latencies)

        return TwinReport(
            total_records=total,
            matched_records=matched,
            diverged_records=diverged,
            match_rate=round(match_rate, 6),
            divergence_points=divergence_points,
            performance_delta_ms=round(avg_mod - avg_orig, 4),
            avg_latency_original_ms=round(avg_orig, 4),
            avg_latency_modified_ms=round(avg_mod, 4),
            generated_at=datetime.datetime.now(datetime.timezone.utc),
        )

    def replay_batch(
        self,
        records: list[ReplayRecord],
        batch_size: int = 100,
    ) -> list[TwinReport]:
        """Replay records in batches and return per-batch reports.

        Parameters
        ----------
        records:
            Full list of replay records.
        batch_size:
            Number of records per batch.

        Returns
        -------
        list[TwinReport]
            One report per batch.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        reports: list[TwinReport] = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            reports.append(self.replay(batch))
        return reports

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _invoke_modified(self, record: ReplayRecord) -> tuple[object, float]:
        """Run the modified agent and measure latency.

        Parameters
        ----------
        record:
            The replay record providing the input.

        Returns
        -------
        tuple[object, float]
            (output, latency_ms)
        """
        start = time.perf_counter()
        try:
            output = self._modified_agent(record.input_data)
        except Exception as exc:
            output = {"error": str(exc)}
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return output, elapsed_ms

    @staticmethod
    def _default_comparator(original: object, modified: object) -> bool:
        """Default comparator — strict equality."""
        return original == modified


__all__ = [
    "DigitalTwin",
    "DivergencePoint",
    "ReplayRecord",
    "TwinReport",
]
