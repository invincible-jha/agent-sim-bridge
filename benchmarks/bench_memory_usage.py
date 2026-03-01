"""Benchmark: Memory usage of gap estimation and scenario management.

Uses tracemalloc to measure peak memory allocated during GapEstimator
construction and repeated multi-metric estimation calls.
"""
from __future__ import annotations

import json
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sim_bridge.gap.estimator import GapDimension, GapEstimator

_ITERATIONS: int = 500


def bench_gap_estimation_memory() -> dict[str, object]:
    """Benchmark memory usage during multi-metric gap estimation.

    Returns
    -------
    dict with keys: operation, iterations, peak_memory_kb, current_memory_kb,
    ops_per_second, avg_latency_ms, memory_peak_mb.
    """
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    estimator = GapEstimator()  # All four metrics.
    dimension = GapDimension(
        name="mem-bench-dim",
        sim_distribution=[0.1, 0.3, 0.4, 0.2],
        real_distribution=[0.15, 0.25, 0.35, 0.25],
    )

    for _ in range(_ITERATIONS):
        estimator.estimate_dimension(dimension)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    peak_kb = round(total_bytes / 1024, 2)

    result: dict[str, object] = {
        "operation": "gap_estimation_memory",
        "iterations": _ITERATIONS,
        "peak_memory_kb": peak_kb,
        "current_memory_kb": peak_kb,
        "ops_per_second": 0.0,
        "avg_latency_ms": 0.0,
        "memory_peak_mb": round(peak_kb / 1024, 4),
    }
    print(
        f"[bench_memory_usage] {result['operation']}: "
        f"peak {peak_kb:.2f} KB over {_ITERATIONS} iterations"
    )
    return result


def run_benchmark() -> dict[str, object]:
    """Entry point returning the benchmark result dict."""
    return bench_gap_estimation_memory()


if __name__ == "__main__":
    result = run_benchmark()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "memory_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Results saved to {output_path}")
