"""Tests for agent_sim_bridge.digital_twin.twin."""
from __future__ import annotations

import pytest

from agent_sim_bridge.digital_twin.twin import (
    DigitalTwin,
    DivergencePoint,
    ReplayRecord,
    TwinReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    record_id: str,
    question: str = "hello",
    original_answer: str = "HELLO",
    latency: float = 50.0,
) -> ReplayRecord:
    return ReplayRecord(
        record_id=record_id,
        input_data={"question": question},
        original_output={"answer": original_answer},
        original_latency_ms=latency,
    )


def _upper_agent(inp: dict) -> dict:
    return {"answer": inp["question"].upper()}


def _lower_agent(inp: dict) -> dict:
    return {"answer": inp["question"].lower()}


def _echo_agent(inp: dict) -> dict:
    return {"answer": inp["question"].upper()}  # Same as original


def _error_agent(inp: dict) -> dict:
    raise RuntimeError("agent crashed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def matching_twin() -> DigitalTwin:
    """Twin where modified agent matches originals."""
    return DigitalTwin(modified_agent=_echo_agent)


@pytest.fixture()
def diverging_twin() -> DigitalTwin:
    """Twin where modified agent always diverges."""
    return DigitalTwin(modified_agent=_lower_agent)


@pytest.fixture()
def records() -> list[ReplayRecord]:
    return [_record(f"r{i}", f"word{i}", f"WORD{i}") for i in range(5)]


# ---------------------------------------------------------------------------
# Empty records
# ---------------------------------------------------------------------------


class TestEmptyReplay:
    def test_empty_records_returns_report(self, matching_twin: DigitalTwin) -> None:
        report = matching_twin.replay([])
        assert isinstance(report, TwinReport)

    def test_empty_match_rate_is_one(self, matching_twin: DigitalTwin) -> None:
        report = matching_twin.replay([])
        assert report.match_rate == 1.0

    def test_empty_total_records_zero(self, matching_twin: DigitalTwin) -> None:
        report = matching_twin.replay([])
        assert report.total_records == 0


# ---------------------------------------------------------------------------
# Full match
# ---------------------------------------------------------------------------


class TestFullMatch:
    def test_all_match_match_rate_one(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        assert report.match_rate == pytest.approx(1.0)

    def test_all_match_no_divergence_points(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        assert report.divergence_points == []

    def test_all_match_matched_count(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        assert report.matched_records == 5
        assert report.diverged_records == 0

    def test_all_match_total_records(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        assert report.total_records == 5


# ---------------------------------------------------------------------------
# Full divergence
# ---------------------------------------------------------------------------


class TestFullDivergence:
    def test_all_diverge_match_rate_zero(
        self, diverging_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = diverging_twin.replay(records)
        assert report.match_rate == pytest.approx(0.0)

    def test_all_diverge_divergence_points_count(
        self, diverging_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = diverging_twin.replay(records)
        assert len(report.divergence_points) == 5

    def test_all_diverge_matched_count(
        self, diverging_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = diverging_twin.replay(records)
        assert report.matched_records == 0
        assert report.diverged_records == 5

    def test_divergence_point_has_both_outputs(
        self, diverging_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = diverging_twin.replay(records)
        for dp in report.divergence_points:
            assert dp.original_output is not None
            assert dp.modified_output is not None

    def test_divergence_point_ids_match_records(
        self, diverging_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = diverging_twin.replay(records)
        record_ids = {r.record_id for r in records}
        dp_ids = {dp.record_id for dp in report.divergence_points}
        assert dp_ids == record_ids


# ---------------------------------------------------------------------------
# Partial divergence
# ---------------------------------------------------------------------------


class TestPartialDivergence:
    def test_partial_match_rate(self) -> None:
        records = [
            _record("match_1", "hello", "HELLO"),
            _record("match_2", "world", "WORLD"),
            _record("diverge_1", "foo", "FOO"),
        ]
        # Modified agent returns upper — matches first two, diverges on "foo"
        # Actually _echo_agent returns upper same as original
        # Use custom agent that flips "foo"
        def _selective(inp: dict) -> dict:
            q = inp["question"]
            if q == "foo":
                return {"answer": "different"}
            return {"answer": q.upper()}

        twin = DigitalTwin(modified_agent=_selective)
        report = twin.replay(records)
        assert report.matched_records == 2
        assert report.diverged_records == 1
        assert report.match_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Performance delta
# ---------------------------------------------------------------------------


class TestPerformanceDelta:
    def test_performance_delta_is_float(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        assert isinstance(report.performance_delta_ms, float)

    def test_avg_latencies_non_negative(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        assert report.avg_latency_original_ms >= 0.0
        assert report.avg_latency_modified_ms >= 0.0

    def test_performance_delta_equals_diff_of_avgs(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        expected = round(
            report.avg_latency_modified_ms - report.avg_latency_original_ms, 4
        )
        assert report.performance_delta_ms == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_agent_exception_captured_as_divergence(self) -> None:
        twin = DigitalTwin(modified_agent=_error_agent)
        records = [_record("e1", "test", "TEST")]
        report = twin.replay(records)
        assert report.diverged_records == 1

    def test_error_output_contains_error_key(self) -> None:
        twin = DigitalTwin(modified_agent=_error_agent)
        records = [_record("e1", "test", "TEST")]
        report = twin.replay(records)
        dp = report.divergence_points[0]
        assert "error" in dp.modified_output


# ---------------------------------------------------------------------------
# Custom comparator
# ---------------------------------------------------------------------------


class TestCustomComparator:
    def test_custom_comparator_used(self) -> None:
        def _always_match(a: object, b: object) -> bool:
            return True

        twin = DigitalTwin(modified_agent=_lower_agent, output_comparator=_always_match)
        records = [_record("r1", "hello", "HELLO")]
        report = twin.replay(records)
        assert report.match_rate == 1.0


# ---------------------------------------------------------------------------
# Batch replay
# ---------------------------------------------------------------------------


class TestBatchReplay:
    def test_batch_replay_returns_multiple_reports(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        reports = matching_twin.replay_batch(records, batch_size=2)
        assert len(reports) == 3  # 2, 2, 1

    def test_batch_replay_total_equals_sum(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        reports = matching_twin.replay_batch(records, batch_size=2)
        total = sum(r.total_records for r in reports)
        assert total == 5

    def test_batch_replay_invalid_batch_size_raises(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        with pytest.raises(ValueError):
            matching_twin.replay_batch(records, batch_size=0)

    def test_report_has_timestamp(
        self, matching_twin: DigitalTwin, records: list[ReplayRecord]
    ) -> None:
        report = matching_twin.replay(records)
        assert report.generated_at is not None
