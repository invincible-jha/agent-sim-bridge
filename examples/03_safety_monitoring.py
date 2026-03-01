#!/usr/bin/env python3
"""Example: Safety Monitoring and Boundary Checking

Demonstrates real-time safety constraint monitoring, boundary
violation detection, and monitored step execution.

Usage:
    python examples/03_safety_monitoring.py

Requirements:
    pip install agent-sim-bridge
"""
from __future__ import annotations

import agent_sim_bridge
from agent_sim_bridge import (
    BoundaryChecker,
    BoundaryDefinition,
    ConstraintType,
    SafetyChecker,
    SafetyConstraint,
    SafetyMonitor,
    ViolationSeverity,
)


def main() -> None:
    print(f"agent-sim-bridge version: {agent_sim_bridge.__version__}")

    # Step 1: Define safety constraints
    constraints = [
        SafetyConstraint(
            name="max-speed",
            constraint_type=ConstraintType.UPPER_BOUND,
            parameter="speed",
            threshold=2.0,
            severity=ViolationSeverity.CRITICAL,
        ),
        SafetyConstraint(
            name="min-battery",
            constraint_type=ConstraintType.LOWER_BOUND,
            parameter="battery_pct",
            threshold=10.0,
            severity=ViolationSeverity.WARNING,
        ),
        SafetyConstraint(
            name="max-temperature",
            constraint_type=ConstraintType.UPPER_BOUND,
            parameter="temperature_c",
            threshold=80.0,
            severity=ViolationSeverity.ERROR,
        ),
    ]
    checker = SafetyChecker(constraints=constraints)

    # Step 2: Check individual observations
    test_states = [
        {"speed": 1.5, "battery_pct": 45.0, "temperature_c": 65.0},
        {"speed": 2.5, "battery_pct": 8.0, "temperature_c": 85.0},
        {"speed": 0.8, "battery_pct": 50.0, "temperature_c": 70.0},
    ]

    print("Safety checks:")
    for state in test_states:
        violations = checker.check_all(state)
        status = "PASS" if not violations else "FAIL"
        print(f"  {status}: speed={state['speed']}, "
              f"battery={state['battery_pct']}%, "
              f"temp={state['temperature_c']}C")
        for v in violations:
            print(f"    [{v.severity.value}] {v.name}: {v.message}")

    # Step 3: Boundary checking (spatial bounds)
    boundary = BoundaryDefinition(
        name="operating-zone",
        x_min=-10.0, x_max=10.0,
        y_min=-10.0, y_max=10.0,
    )
    boundary_checker = BoundaryChecker(boundaries=[boundary])

    positions = [
        {"x": 5.0, "y": 3.0},
        {"x": 12.0, "y": 0.0},   # out of bounds
        {"x": -5.0, "y": -11.0}, # out of bounds
    ]
    print("\nBoundary checks:")
    for pos in positions:
        bv = boundary_checker.check(pos)
        in_bounds = bv is None
        print(f"  ({pos['x']:+.1f}, {pos['y']:+.1f}): "
              f"{'in-bounds' if in_bounds else f'VIOLATION: {bv.name}'}")

    # Step 4: Safety monitor (aggregated)
    monitor = SafetyMonitor(checker=checker)
    for state in test_states:
        monitor.observe(state)
    report = monitor.report()
    print(f"\nMonitor report: {report.total_observations} observations, "
          f"{report.violation_count} violations, "
          f"pass_rate={report.pass_rate:.0%}")


if __name__ == "__main__":
    main()
