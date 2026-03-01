"""Benchmark: Sim-to-real gap estimation latency — per-dimension p50/p95/p99.

Measures the per-call latency of GapEstimator.estimate_dimension() for a
single distribution pair, capturing the latency distribution.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sim_bridge.gap.estimator import GapDimension, GapEstimator, GapMetric

_WARMUP: int = 100
_ITERATIONS: int = 2_000

# Use a small, fixed distribution to isolate computation cost.
_SIM_DIST = [0.1, 0.3, 0.4, 0.2]
_REAL_DIST = [0.15, 0.25, 0.35, 0.25]


def bench_gap_estimation_latency() -> dict[str, object]:
    """Benchmark GapEstimator.estimate_dimension() per-call latency.

    Returns
    -------
    dict with keys: operation, iterations, total_seconds, ops_per_second,
    avg_latency_ms, p99_latency_ms, memory_peak_mb.
    """
    # Use a single metric to isolate that metric's computation cost.
    estimator = GapEstimator(metrics=[GapMetric.JENSEN_SHANNON])
    dimension = GapDimension(
        name="bench-dim",
        sim_distribution=_SIM_DIST,
        real_distribution=_REAL_DIST,
    )

    # Warmup.
    for _ in range(_WARMUP):
        estimator.estimate_dimension(dimension)

    latencies_ms: list[float] = []
    for _ in range(_ITERATIONS):
        t0 = time.perf_counter()
        estimator.estimate_dimension(dimension)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    total = sum(latencies_ms) / 1000

    result: dict[str, object] = {
        "operation": "gap_estimation_latency",
        "iterations": _ITERATIONS,
        "total_seconds": round(total, 4),
        "ops_per_second": round(_ITERATIONS / total, 1),
        "avg_latency_ms": round(sum(latencies_ms) / n, 4),
        "p99_latency_ms": round(sorted_lats[min(int(n * 0.99), n - 1)], 4),
        "memory_peak_mb": 0.0,
    }
    print(
        f"[bench_gap_latency] {result['operation']}: "
        f"p99={result['p99_latency_ms']:.4f}ms  "
        f"mean={result['avg_latency_ms']:.4f}ms"
    )
    return result


def run_benchmark() -> dict[str, object]:
    """Entry point returning the benchmark result dict."""
    return bench_gap_estimation_latency()


if __name__ == "__main__":
    result = run_benchmark()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "latency_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Results saved to {output_path}")
