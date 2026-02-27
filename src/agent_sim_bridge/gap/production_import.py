"""Production telemetry importer for sim-to-real gap estimation.

Imports production telemetry in JSON Lines or CSV format and compares
real-world metrics against simulated metrics to quantify the sim-to-real
gap.

Supported formats
-----------------
- **JSON Lines** (``.jsonl``): one JSON object per line, each representing a
  single observation with metric name → value mappings.
- **CSV** (``.csv``): first row is a header; each subsequent row is one
  observation.

The importer produces :class:`GapDimension` objects compatible with the
existing :class:`~agent_sim_bridge.gap.estimator.GapEstimator`.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

from agent_sim_bridge.gap.estimator import GapDimension


# ---------------------------------------------------------------------------
# Import result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportedTelemetry:
    """Container for telemetry data loaded from a file or string.

    Attributes
    ----------
    source_name:
        Identifier for the data source (filename, URL, etc.).
    records:
        List of dicts — each dict maps metric name → value.
    record_count:
        Number of successfully parsed records.
    metric_names:
        Set of metric names found across all records.
    parse_errors:
        Number of lines/rows that failed to parse.
    """

    source_name: str
    records: list[dict[str, object]]
    record_count: int
    metric_names: set[str]
    parse_errors: int


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricComparison:
    """Comparison of a single metric between real and simulated data.

    Attributes
    ----------
    metric_name:
        The metric being compared.
    real_mean:
        Mean of the real-world values.
    sim_mean:
        Mean of the simulated values.
    mean_delta:
        Absolute difference: ``abs(real_mean - sim_mean)``.
    mean_delta_pct:
        Relative difference as a percentage of real_mean.
    real_std:
        Standard deviation of real-world values.
    sim_std:
        Standard deviation of simulated values.
    sample_count_real:
        Number of real-world samples.
    sample_count_sim:
        Number of simulated samples.
    gap_dimension:
        A :class:`GapDimension` ready for use with :class:`GapEstimator`.
    """

    metric_name: str
    real_mean: float
    sim_mean: float
    mean_delta: float
    mean_delta_pct: float
    real_std: float
    sim_std: float
    sample_count_real: int
    sample_count_sim: int
    gap_dimension: GapDimension


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


class ProductionTelemetryImporter:
    """Import and parse production telemetry for gap estimation.

    Example
    -------
    ::

        importer = ProductionTelemetryImporter()
        telemetry = importer.load_jsonl(jsonl_string, source_name="prod")
        sim_data = {"latency_ms": [100, 120, 110], "error_rate": [0.01, 0.02]}
        comparisons = importer.compare(telemetry, sim_data)
    """

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_jsonl(
        self, content: str, source_name: str = "unknown"
    ) -> ImportedTelemetry:
        """Parse JSON Lines format telemetry.

        Parameters
        ----------
        content:
            The raw JSON Lines string (one JSON object per line).
        source_name:
            Label for this data source.

        Returns
        -------
        ImportedTelemetry
            Parsed telemetry container.
        """
        records: list[dict[str, object]] = []
        errors = 0
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
                else:
                    errors += 1
            except json.JSONDecodeError:
                errors += 1

        metric_names: set[str] = set()
        for record in records:
            metric_names.update(k for k, v in record.items() if isinstance(v, (int, float)))

        return ImportedTelemetry(
            source_name=source_name,
            records=records,
            record_count=len(records),
            metric_names=metric_names,
            parse_errors=errors,
        )

    def load_csv(
        self, content: str, source_name: str = "unknown"
    ) -> ImportedTelemetry:
        """Parse CSV format telemetry.

        Parameters
        ----------
        content:
            The raw CSV string (first row = header).
        source_name:
            Label for this data source.

        Returns
        -------
        ImportedTelemetry
            Parsed telemetry container.
        """
        records: list[dict[str, object]] = []
        errors = 0
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            parsed_row: dict[str, object] = {}
            row_ok = True
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed_row[key] = float(value)
                except (ValueError, TypeError):
                    parsed_row[key] = value  # Keep as string if not numeric
            if row_ok:
                records.append(parsed_row)
            else:
                errors += 1

        metric_names: set[str] = set()
        for record in records:
            metric_names.update(k for k, v in record.items() if isinstance(v, float))

        return ImportedTelemetry(
            source_name=source_name,
            records=records,
            record_count=len(records),
            metric_names=metric_names,
            parse_errors=errors,
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        real_telemetry: ImportedTelemetry,
        sim_data: dict[str, list[float]],
    ) -> list[MetricComparison]:
        """Compare real telemetry against simulated data.

        For each metric in *sim_data* that also appears in *real_telemetry*,
        compute descriptive statistics and return a :class:`MetricComparison`.

        Parameters
        ----------
        real_telemetry:
            Imported real-world telemetry.
        sim_data:
            Mapping of metric name → list of simulated values.

        Returns
        -------
        list[MetricComparison]
            One comparison per matched metric.
        """
        comparisons: list[MetricComparison] = []

        # Extract real values per metric
        real_values: dict[str, list[float]] = {}
        for record in real_telemetry.records:
            for metric, value in record.items():
                if isinstance(value, (int, float)):
                    real_values.setdefault(metric, []).append(float(value))

        for metric_name, sim_vals in sim_data.items():
            real_vals = real_values.get(metric_name, [])
            if not real_vals or not sim_vals:
                continue

            real_mean = self._mean(real_vals)
            sim_mean = self._mean(sim_vals)
            mean_delta = abs(real_mean - sim_mean)
            mean_delta_pct = (mean_delta / abs(real_mean) * 100.0) if real_mean != 0.0 else 0.0
            real_std = self._std(real_vals)
            sim_std = self._std(sim_vals)

            gap_dim = GapDimension(
                name=metric_name,
                sim_distribution=sim_vals,
                real_distribution=real_vals,
            )

            comparisons.append(
                MetricComparison(
                    metric_name=metric_name,
                    real_mean=real_mean,
                    sim_mean=sim_mean,
                    mean_delta=mean_delta,
                    mean_delta_pct=mean_delta_pct,
                    real_std=real_std,
                    sim_std=sim_std,
                    sample_count_real=len(real_vals),
                    sample_count_sim=len(sim_vals),
                    gap_dimension=gap_dim,
                )
            )

        return comparisons

    def extract_metric(
        self, telemetry: ImportedTelemetry, metric_name: str
    ) -> list[float]:
        """Extract all numeric values for a single metric from telemetry.

        Parameters
        ----------
        telemetry:
            The imported telemetry.
        metric_name:
            The metric key to extract.

        Returns
        -------
        list[float]
            All numeric values found for the metric.
        """
        values: list[float] = []
        for record in telemetry.records:
            v = record.get(metric_name)
            if isinstance(v, (int, float)):
                values.append(float(v))
        return values

    # ------------------------------------------------------------------
    # Private statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Compute the arithmetic mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std(values: list[float]) -> float:
        """Compute population standard deviation."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return variance ** 0.5


__all__ = [
    "ImportedTelemetry",
    "MetricComparison",
    "ProductionTelemetryImporter",
]
