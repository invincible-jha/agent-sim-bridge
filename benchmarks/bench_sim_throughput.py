"""Benchmark: Simulation scenario evaluation throughput — scenarios per second.

Measures how many Scenario.evaluate() calls can be completed per second
using the built-in pass/fail criteria logic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sim_bridge.simulation.scenario import Scenario, ScenarioOutcomeCriteria

_ITERATIONS: int = 10_000


def _make_scenario() -> Scenario:
    """Build a scenario with success criteria for benchmarking."""
    return Scenario(
        name="bench-scenario",
        description="Throughput benchmark scenario",
        seed=42,
        outcome=ScenarioOutcomeCriteria(
            min_reward=10.0,
            max_steps=100,
            require_termination=True,
        ),
    )


def bench_scenario_evaluation_throughput() -> dict[str, object]:
    """Benchmark Scenario.evaluate() throughput.

    Returns
    -------
    dict with keys: operation, iterations, total_seconds, ops_per_second,
    avg_latency_ms, p99_latency_ms, memory_peak_mb.
    """
    scenario = _make_scenario()

    start = time.perf_counter()
    for i in range(_ITERATIONS):
        # Alternate between passing and failing to avoid branch prediction optimisation.
        reward = 15.0 if i % 2 == 0 else 5.0
        scenario.evaluate(total_reward=reward, steps=50, terminated=True)
    total = time.perf_counter() - start

    result: dict[str, object] = {
        "operation": "scenario_evaluation_throughput",
        "iterations": _ITERATIONS,
        "total_seconds": round(total, 4),
        "ops_per_second": round(_ITERATIONS / total, 1),
        "avg_latency_ms": round(total / _ITERATIONS * 1000, 4),
        "p99_latency_ms": 0.0,
        "memory_peak_mb": 0.0,
    }
    print(
        f"[bench_sim_throughput] {result['operation']}: "
        f"{result['ops_per_second']:,.0f} ops/sec  "
        f"avg {result['avg_latency_ms']:.4f} ms"
    )
    return result


def run_benchmark() -> dict[str, object]:
    """Entry point returning the benchmark result dict."""
    return bench_scenario_evaluation_throughput()


if __name__ == "__main__":
    result = run_benchmark()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "throughput_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Results saved to {output_path}")
