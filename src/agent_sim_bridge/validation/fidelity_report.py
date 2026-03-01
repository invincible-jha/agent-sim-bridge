"""FidelityReport — structured comparison of sim vs real environment outputs.

A FidelityReport is the top-level artefact produced by a ValidationHarness
run.  It aggregates per-scenario results into an overall fidelity score and
provides both machine-readable (dict) and human-readable (markdown) output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_sim_bridge.validation.environment import EnvironmentOutput


@dataclass
class ScenarioResult:
    """Comparison results for a single validation scenario.

    Attributes
    ----------
    scenario_id:
        Identifier of the ValidationScenario that produced this result.
    scenario_name:
        Human-readable name of the scenario.
    sim_outputs:
        Outputs collected from the simulation environment.
    real_outputs:
        Outputs collected from the production/real environment.
    agreement_score:
        Text-similarity score between sim and real responses (0.0–1.0).
        1.0 means identical outputs; 0.0 means no overlap.
    latency_ratio:
        Ratio of total sim latency to total real latency.  Values < 1.0
        indicate the simulation is faster than reality.
    details:
        Arbitrary additional data recorded during comparison.
    """

    scenario_id: str
    scenario_name: str
    sim_outputs: list[EnvironmentOutput]
    real_outputs: list[EnvironmentOutput]
    agreement_score: float
    latency_ratio: float
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise this result to a plain dict."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "agreement_score": self.agreement_score,
            "latency_ratio": self.latency_ratio,
            "n_sim_outputs": len(self.sim_outputs),
            "n_real_outputs": len(self.real_outputs),
            "details": self.details,
        }


@dataclass
class FidelityReport:
    """Transfer fidelity report comparing sim vs real environment outputs.

    Aggregates ScenarioResult objects and exposes summary statistics.

    Attributes
    ----------
    timestamp:
        ISO-8601 UTC timestamp recorded when the report was created.
    sim_environment:
        Name of the simulation environment.
    real_environment:
        Name of the real/production environment.
    scenarios:
        List of per-scenario comparison results.
    """

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sim_environment: str = ""
    real_environment: str = ""
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def overall_fidelity(self) -> float:
        """Mean agreement score across all scenarios (0.0–1.0).

        Returns 0.0 if no scenarios have been recorded.
        """
        if not self.scenarios:
            return 0.0
        total = sum(scenario.agreement_score for scenario in self.scenarios)
        return total / len(self.scenarios)

    @property
    def overall_latency_ratio(self) -> float:
        """Mean latency ratio (sim / real) across all scenarios.

        Returns 0.0 if no scenarios have been recorded.
        """
        if not self.scenarios:
            return 0.0
        total = sum(scenario.latency_ratio for scenario in self.scenarios)
        return total / len(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        """Serialise the full report to a nested plain dict.

        Returns
        -------
        dict[str, object]
            A JSON-serialisable representation of the report.
        """
        return {
            "timestamp": self.timestamp,
            "sim_environment": self.sim_environment,
            "real_environment": self.real_environment,
            "overall_fidelity": self.overall_fidelity,
            "overall_latency_ratio": self.overall_latency_ratio,
            "n_scenarios": len(self.scenarios),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    def to_markdown(self) -> str:
        """Generate a human-readable markdown summary of the report.

        Returns
        -------
        str
            Markdown-formatted text suitable for display in a terminal or
            rendered in a documentation page.
        """
        lines: list[str] = [
            "# Sim-to-Real Fidelity Report",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Simulation environment:** {self.sim_environment or '(unset)'}",
            f"**Real environment:** {self.real_environment or '(unset)'}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall fidelity | {self.overall_fidelity:.4f} |",
            f"| Overall latency ratio (sim/real) | {self.overall_latency_ratio:.4f} |",
            f"| Scenarios evaluated | {len(self.scenarios)} |",
            "",
        ]

        if self.scenarios:
            lines += [
                "## Per-Scenario Results",
                "",
                "| Scenario | Category/ID | Agreement | Latency Ratio |",
                "|----------|-------------|-----------|---------------|",
            ]
            for scenario_result in self.scenarios:
                lines.append(
                    f"| {scenario_result.scenario_name} "
                    f"| {scenario_result.scenario_id} "
                    f"| {scenario_result.agreement_score:.4f} "
                    f"| {scenario_result.latency_ratio:.4f} |"
                )
            lines.append("")

        fidelity_pct = self.overall_fidelity * 100.0
        if fidelity_pct >= 80.0:
            verdict = "PASS — high sim-to-real fidelity"
        elif fidelity_pct >= 50.0:
            verdict = "WARN — moderate fidelity, calibration recommended"
        else:
            verdict = "FAIL — low fidelity, significant sim-to-real gap detected"

        lines += [
            "## Verdict",
            "",
            f"**{verdict}** (overall fidelity {fidelity_pct:.1f}%)",
            "",
        ]

        return "\n".join(lines)
