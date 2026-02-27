"""Tests for agent_sim_bridge.gap.production_import."""
from __future__ import annotations

import json

import pytest

from agent_sim_bridge.gap.production_import import (
    ImportedTelemetry,
    MetricComparison,
    ProductionTelemetryImporter,
)
from agent_sim_bridge.gap.estimator import GapDimension


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def importer() -> ProductionTelemetryImporter:
    return ProductionTelemetryImporter()


JSONL_CONTENT = "\n".join(
    json.dumps({"latency_ms": 100 + i, "error_rate": 0.01 * i, "label": "ok"})
    for i in range(10)
)

CSV_CONTENT = "latency_ms,error_rate\n" + "\n".join(
    f"{100 + i},{0.01 * i}" for i in range(10)
)

SIM_DATA: dict[str, list[float]] = {
    "latency_ms": [110.0, 115.0, 112.0, 108.0, 120.0],
    "error_rate": [0.02, 0.03, 0.025, 0.015, 0.01],
}


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


class TestJsonlLoading:
    def test_load_jsonl_record_count(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT, "test_source")
        assert tel.record_count == 10

    def test_load_jsonl_source_name(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT, "my_source")
        assert tel.source_name == "my_source"

    def test_load_jsonl_metric_names(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        assert "latency_ms" in tel.metric_names
        assert "error_rate" in tel.metric_names
        # String field "label" should not appear in metric_names
        assert "label" not in tel.metric_names

    def test_load_jsonl_zero_errors(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        assert tel.parse_errors == 0

    def test_load_jsonl_invalid_line_counted(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        content = JSONL_CONTENT + "\nnot-valid-json"
        tel = importer.load_jsonl(content)
        assert tel.parse_errors == 1

    def test_load_jsonl_empty_string(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl("")
        assert tel.record_count == 0
        assert tel.metric_names == set()

    def test_load_jsonl_skips_blank_lines(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        content = '{"a": 1}\n\n{"b": 2}\n'
        tel = importer.load_jsonl(content)
        assert tel.record_count == 2


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


class TestCsvLoading:
    def test_load_csv_record_count(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_csv(CSV_CONTENT, "csv_source")
        assert tel.record_count == 10

    def test_load_csv_metric_names(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_csv(CSV_CONTENT)
        assert "latency_ms" in tel.metric_names
        assert "error_rate" in tel.metric_names

    def test_load_csv_source_name(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_csv(CSV_CONTENT, "csv_source")
        assert tel.source_name == "csv_source"

    def test_load_csv_values_numeric(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_csv(CSV_CONTENT)
        vals = importer.extract_metric(tel, "latency_ms")
        assert all(isinstance(v, float) for v in vals)

    def test_load_csv_empty(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_csv("latency_ms,error_rate\n")
        assert tel.record_count == 0


# ---------------------------------------------------------------------------
# Extract metric
# ---------------------------------------------------------------------------


class TestExtractMetric:
    def test_extract_known_metric(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        vals = importer.extract_metric(tel, "latency_ms")
        assert len(vals) == 10

    def test_extract_unknown_metric_empty(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        vals = importer.extract_metric(tel, "nonexistent")
        assert vals == []

    def test_extract_string_field_excluded(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        vals = importer.extract_metric(tel, "label")
        assert vals == []


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_compare_returns_list(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        comparisons = importer.compare(tel, SIM_DATA)
        assert isinstance(comparisons, list)

    def test_compare_matched_metrics(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        comparisons = importer.compare(tel, SIM_DATA)
        names = {c.metric_name for c in comparisons}
        assert "latency_ms" in names
        assert "error_rate" in names

    def test_compare_mean_delta_non_negative(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        for cmp in importer.compare(tel, SIM_DATA):
            assert cmp.mean_delta >= 0.0

    def test_compare_sample_counts(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        for cmp in importer.compare(tel, SIM_DATA):
            assert cmp.sample_count_real == 10
            assert cmp.sample_count_sim == 5

    def test_compare_gap_dimension_type(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        for cmp in importer.compare(tel, SIM_DATA):
            assert isinstance(cmp.gap_dimension, GapDimension)

    def test_compare_gap_dimension_name_matches(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        for cmp in importer.compare(tel, SIM_DATA):
            assert cmp.gap_dimension.name == cmp.metric_name

    def test_compare_unmatched_sim_metric_excluded(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        extended_sim = dict(SIM_DATA)
        extended_sim["nonexistent_metric"] = [1.0, 2.0]
        comparisons = importer.compare(tel, extended_sim)
        names = {c.metric_name for c in comparisons}
        assert "nonexistent_metric" not in names

    def test_compare_std_non_negative(self, importer: ProductionTelemetryImporter) -> None:
        tel = importer.load_jsonl(JSONL_CONTENT)
        for cmp in importer.compare(tel, SIM_DATA):
            assert cmp.real_std >= 0.0
            assert cmp.sim_std >= 0.0

    def test_compare_mean_delta_pct_zero_when_means_equal(
        self, importer: ProductionTelemetryImporter
    ) -> None:
        content = "\n".join(json.dumps({"x": 5.0}) for _ in range(5))
        tel = importer.load_jsonl(content)
        comparisons = importer.compare(tel, {"x": [5.0, 5.0, 5.0]})
        assert comparisons[0].mean_delta == pytest.approx(0.0)
        assert comparisons[0].mean_delta_pct == pytest.approx(0.0)
