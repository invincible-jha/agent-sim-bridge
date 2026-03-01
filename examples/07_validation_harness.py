#!/usr/bin/env python3
"""Example: Validation Harness and Fidelity Reporting

Demonstrates running standard validation scenarios against a mock
environment and generating a fidelity report.

Usage:
    python examples/07_validation_harness.py

Requirements:
    pip install agent-sim-bridge
"""
from __future__ import annotations

import agent_sim_bridge
from agent_sim_bridge import (
    FidelityReport,
    MockEnvironment,
    STANDARD_SCENARIOS,
    ScenarioResult,
    ValidationHarness,
    ValidationScenario,
)


def main() -> None:
    print(f"agent-sim-bridge version: {agent_sim_bridge.__version__}")

    # Step 1: Inspect standard scenarios
    print(f"Standard scenarios ({len(STANDARD_SCENARIOS)}):")
    for scenario in STANDARD_SCENARIOS[:3]:
        print(f"  [{scenario.scenario_id}] {scenario.name}: "
              f"{scenario.description[:50]}")

    # Step 2: Create a mock environment for validation
    mock_env = MockEnvironment(
        fidelity=0.85,  # 85% fidelity to reality
        name="warehouse-mock",
    )

    # Step 3: Run validation harness
    harness = ValidationHarness(environment=mock_env)
    results: list[ScenarioResult] = harness.run(
        scenarios=STANDARD_SCENARIOS[:3]
    )

    print(f"\nValidation results ({len(results)} scenarios):")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.scenario_id}: "
              f"fidelity={result.fidelity_score:.2f}, "
              f"steps={result.steps_taken}")

    # Step 4: Custom validation scenario
    custom_scenario = ValidationScenario(
        scenario_id="custom-recovery",
        name="Obstacle Recovery",
        description="Agent recovers from unexpected obstacle.",
        max_steps=20,
        success_criteria={"reached_goal": True, "no_collisions": True},
    )
    custom_result = harness.run_single(custom_scenario)
    print(f"\nCustom scenario: {custom_scenario.name}")
    print(f"  Passed: {custom_result.passed}")
    print(f"  Fidelity: {custom_result.fidelity_score:.2f}")

    # Step 5: Fidelity report
    report: FidelityReport = harness.fidelity_report(results + [custom_result])
    print(f"\nFidelity report:")
    print(f"  Total scenarios: {report.total_scenarios}")
    print(f"  Passed: {report.passed_count}")
    print(f"  Pass rate: {report.pass_rate:.0%}")
    print(f"  Avg fidelity: {report.avg_fidelity:.3f}")
    print(f"  Overall grade: {report.grade}")


if __name__ == "__main__":
    main()
