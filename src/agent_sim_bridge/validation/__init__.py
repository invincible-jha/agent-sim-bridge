"""Validation subsystem — sim-to-real transfer validation harness.

Provides:

* **environment** — :class:`Environment` ABC, :class:`EnvironmentInput`,
  :class:`EnvironmentOutput`, and :class:`MockEnvironment` for testing.
* **scenarios** — :class:`ValidationScenario` and the five
  :data:`STANDARD_SCENARIOS` covering classification, extraction, QA,
  summarization, and multi-step reasoning.
* **fidelity_report** — :class:`FidelityReport` and :class:`ScenarioResult`
  for structured output of validation runs.
* **harness** — :class:`ValidationHarness` that drives paired environments
  through scenarios and produces a :class:`FidelityReport`.
"""
from __future__ import annotations

from agent_sim_bridge.validation.environment import (
    Environment,
    EnvironmentInput,
    EnvironmentOutput,
    MockEnvironment,
)
from agent_sim_bridge.validation.fidelity_report import FidelityReport, ScenarioResult
from agent_sim_bridge.validation.harness import ValidationHarness
from agent_sim_bridge.validation.scenarios import STANDARD_SCENARIOS, ValidationScenario

__all__ = [
    # environment
    "Environment",
    "EnvironmentInput",
    "EnvironmentOutput",
    "MockEnvironment",
    # scenarios
    "ValidationScenario",
    "STANDARD_SCENARIOS",
    # fidelity_report
    "ScenarioResult",
    "FidelityReport",
    # harness
    "ValidationHarness",
]
