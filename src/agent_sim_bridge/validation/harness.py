"""ValidationHarness — orchestrates sim-to-real validation runs.

The harness drives a paired set of environments (simulation and real)
through a collection of ValidationScenarios, compares their outputs,
and assembles the results into a FidelityReport.
"""
from __future__ import annotations

import logging
from difflib import SequenceMatcher

from agent_sim_bridge.validation.environment import (
    Environment,
    EnvironmentOutput,
)
from agent_sim_bridge.validation.fidelity_report import FidelityReport, ScenarioResult
from agent_sim_bridge.validation.scenarios import STANDARD_SCENARIOS, ValidationScenario

logger = logging.getLogger(__name__)


def _text_similarity(text_a: str, text_b: str) -> float:
    """Return a similarity ratio between two strings using SequenceMatcher.

    Parameters
    ----------
    text_a:
        First string.
    text_b:
        Second string.

    Returns
    -------
    float
        Similarity ratio in [0.0, 1.0].  1.0 = identical.
    """
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


class ValidationHarness:
    """Orchestrates sim-to-real validation across multiple scenarios.

    Runs each ValidationScenario against both a simulation and a real
    environment, computes agreement and latency metrics, and produces a
    FidelityReport.

    Parameters
    ----------
    sim_env:
        The simulation environment to validate against.
    real_env:
        The real/production environment to compare to.

    Example
    -------
    ::

        harness = ValidationHarness(sim_env=my_sim, real_env=my_real)
        report = harness.run_all()
        print(report.to_markdown())
    """

    def __init__(self, sim_env: Environment, real_env: Environment) -> None:
        self._sim = sim_env
        self._real = real_env

    @property
    def sim_environment(self) -> Environment:
        """The simulation environment under test."""
        return self._sim

    @property
    def real_environment(self) -> Environment:
        """The real/production environment under test."""
        return self._real

    def run_scenario(self, scenario: ValidationScenario) -> ScenarioResult:
        """Run one scenario in both environments and compute comparison metrics.

        Each input in the scenario is sent to the sim environment first, then
        to the real environment.  Outputs are paired positionally.

        Parameters
        ----------
        scenario:
            The scenario to execute.

        Returns
        -------
        ScenarioResult
            Paired outputs plus agreement and latency metrics.
        """
        sim_outputs: list[EnvironmentOutput] = []
        real_outputs: list[EnvironmentOutput] = []

        for input_data in scenario.inputs:
            sim_out = self._sim.execute(input_data)
            real_out = self._real.execute(input_data)
            sim_outputs.append(sim_out)
            real_outputs.append(real_out)
            logger.debug(
                "Scenario %r, prompt=%r: sim=%r, real=%r",
                scenario.id,
                input_data.prompt[:40],
                sim_out.response[:40],
                real_out.response[:40],
            )

        agreement = self._compute_agreement(sim_outputs, real_outputs)
        latency_ratio = self._compute_latency_ratio(sim_outputs, real_outputs)

        logger.info(
            "Scenario %r complete: agreement=%.4f, latency_ratio=%.4f",
            scenario.id,
            agreement,
            latency_ratio,
        )

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            sim_outputs=sim_outputs,
            real_outputs=real_outputs,
            agreement_score=agreement,
            latency_ratio=latency_ratio,
        )

    def run_all(
        self,
        scenarios: list[ValidationScenario] | None = None,
    ) -> FidelityReport:
        """Run all scenarios and produce a consolidated FidelityReport.

        Parameters
        ----------
        scenarios:
            Explicit list of scenarios to run.  Defaults to
            :data:`~agent_sim_bridge.validation.scenarios.STANDARD_SCENARIOS`
            when ``None``.

        Returns
        -------
        FidelityReport
            Aggregated report across all scenarios.
        """
        resolved_scenarios = STANDARD_SCENARIOS if scenarios is None else scenarios

        report = FidelityReport(
            sim_environment=self._sim.name,
            real_environment=self._real.name,
        )

        for scenario in resolved_scenarios:
            result = self.run_scenario(scenario)
            report.scenarios.append(result)

        logger.info(
            "Validation complete: %d scenarios, overall_fidelity=%.4f",
            len(report.scenarios),
            report.overall_fidelity,
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_agreement(
        self,
        sim_outputs: list[EnvironmentOutput],
        real_outputs: list[EnvironmentOutput],
    ) -> float:
        """Compute mean text-similarity agreement between paired outputs.

        Parameters
        ----------
        sim_outputs:
            Outputs from the simulation environment.
        real_outputs:
            Corresponding outputs from the real environment.

        Returns
        -------
        float
            Mean similarity ratio in [0.0, 1.0].  Returns 0.0 if either
            list is empty.
        """
        if not sim_outputs or not real_outputs:
            return 0.0

        scores: list[float] = []
        for sim_out, real_out in zip(sim_outputs, real_outputs):
            scores.append(_text_similarity(sim_out.response, real_out.response))

        return sum(scores) / len(scores)

    def _compute_latency_ratio(
        self,
        sim_outputs: list[EnvironmentOutput],
        real_outputs: list[EnvironmentOutput],
    ) -> float:
        """Compute the ratio of total sim latency to total real latency.

        Parameters
        ----------
        sim_outputs:
            Outputs from the simulation environment.
        real_outputs:
            Corresponding outputs from the real environment.

        Returns
        -------
        float
            Ratio (sim_total / real_total).  Returns 0.0 if real latency
            sums to zero to avoid division by zero.
        """
        total_sim = sum(output.latency_seconds for output in sim_outputs)
        total_real = sum(output.latency_seconds for output in real_outputs)
        if total_real == 0.0:
            return 0.0
        return total_sim / total_real

    def __repr__(self) -> str:
        return (
            f"ValidationHarness("
            f"sim={self._sim.name!r}, "
            f"real={self._real.name!r})"
        )
