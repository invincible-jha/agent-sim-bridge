"""Tests for the sim-to-real validation harness.

Covers:
- MockEnvironment creation, execute, reset
- ValidationScenario creation and STANDARD_SCENARIOS
- ValidationHarness.run_scenario (matching and differing outputs)
- ValidationHarness.run_all
- FidelityReport: overall_fidelity, overall_latency_ratio, to_dict, to_markdown
- ScenarioResult: creation, fields
- Agreement computation: identical, different, empty
- Environment ABC enforcement
- Integration: full sim->real->report pipeline
"""
from __future__ import annotations

import pytest

from agent_sim_bridge.validation.environment import (
    Environment,
    EnvironmentInput,
    EnvironmentOutput,
    MockEnvironment,
)
from agent_sim_bridge.validation.fidelity_report import FidelityReport, ScenarioResult
from agent_sim_bridge.validation.harness import ValidationHarness, _text_similarity
from agent_sim_bridge.validation.scenarios import STANDARD_SCENARIOS, ValidationScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim_env(responses: dict[str, str] | None = None) -> MockEnvironment:
    return MockEnvironment(name="test-sim", env_type="simulation", responses=responses)


def _make_real_env(responses: dict[str, str] | None = None) -> MockEnvironment:
    return MockEnvironment(
        name="test-real",
        env_type="production",
        responses=responses,
        fixed_latency=0.1,
    )


def _make_scenario(
    prompts: list[str],
    scenario_id: str = "test_scenario",
    category: str = "qa",
) -> ValidationScenario:
    inputs = [
        EnvironmentInput(prompt=p, scenario_id=scenario_id) for p in prompts
    ]
    return ValidationScenario(
        id=scenario_id,
        name="Test Scenario",
        description="A test scenario",
        category=category,
        inputs=inputs,
        expected_properties={"key": "value"},
    )


# ---------------------------------------------------------------------------
# MockEnvironment tests
# ---------------------------------------------------------------------------


class TestMockEnvironment:
    def test_creation_defaults(self) -> None:
        env = MockEnvironment()
        assert env.name == "mock"
        assert env.environment_type == "simulation"
        assert env.call_count == 0

    def test_creation_custom_params(self) -> None:
        env = MockEnvironment(
            name="my-sim",
            env_type="production",
            responses={"hello": "world"},
            fixed_latency=0.05,
        )
        assert env.name == "my-sim"
        assert env.environment_type == "production"

    def test_execute_returns_output(self) -> None:
        env = MockEnvironment()
        input_data = EnvironmentInput(prompt="test prompt")
        output = env.execute(input_data)
        assert isinstance(output, EnvironmentOutput)
        assert "test prompt" in output.response
        assert output.latency_seconds == 0.001

    def test_execute_uses_response_lookup(self) -> None:
        env = MockEnvironment(responses={"hello": "world response"})
        input_data = EnvironmentInput(prompt="hello")
        output = env.execute(input_data)
        assert output.response == "world response"

    def test_execute_increments_call_count(self) -> None:
        env = MockEnvironment()
        assert env.call_count == 0
        env.execute(EnvironmentInput(prompt="p1"))
        env.execute(EnvironmentInput(prompt="p2"))
        assert env.call_count == 2

    def test_reset_clears_call_count(self) -> None:
        env = MockEnvironment()
        env.execute(EnvironmentInput(prompt="p1"))
        env.execute(EnvironmentInput(prompt="p2"))
        env.reset()
        assert env.call_count == 0

    def test_fallback_response_when_prompt_not_in_lookup(self) -> None:
        env = MockEnvironment(responses={"known": "answer"})
        output = env.execute(EnvironmentInput(prompt="unknown prompt"))
        assert "unknown prompt" in output.response

    def test_repr_includes_name(self) -> None:
        env = MockEnvironment(name="my-env")
        assert "my-env" in repr(env)


# ---------------------------------------------------------------------------
# EnvironmentInput / EnvironmentOutput tests
# ---------------------------------------------------------------------------


class TestEnvironmentValueObjects:
    def test_environment_input_is_frozen(self) -> None:
        inp = EnvironmentInput(prompt="hello", context={"k": "v"}, scenario_id="s1")
        with pytest.raises((AttributeError, TypeError)):
            inp.prompt = "changed"  # type: ignore[misc]

    def test_environment_output_is_frozen(self) -> None:
        out = EnvironmentOutput(response="r", latency_seconds=0.01)
        with pytest.raises((AttributeError, TypeError)):
            out.response = "changed"  # type: ignore[misc]

    def test_environment_output_has_timestamp(self) -> None:
        out = EnvironmentOutput(response="r", latency_seconds=0.01)
        assert out.timestamp  # non-empty
        assert "T" in out.timestamp  # ISO-8601

    def test_environment_input_defaults(self) -> None:
        inp = EnvironmentInput(prompt="test")
        assert inp.context == {}
        assert inp.scenario_id == ""


# ---------------------------------------------------------------------------
# ValidationScenario tests
# ---------------------------------------------------------------------------


class TestValidationScenario:
    def test_scenario_creation(self) -> None:
        scenario = _make_scenario(["prompt1", "prompt2"], scenario_id="s1")
        assert scenario.id == "s1"
        assert scenario.name == "Test Scenario"
        assert len(scenario.inputs) == 2
        assert scenario.category == "qa"
        assert scenario.expected_properties == {"key": "value"}

    def test_scenario_is_frozen(self) -> None:
        scenario = _make_scenario(["p"])
        with pytest.raises((AttributeError, TypeError)):
            scenario.id = "changed"  # type: ignore[misc]

    def test_standard_scenarios_count(self) -> None:
        assert len(STANDARD_SCENARIOS) == 5

    def test_standard_scenarios_have_three_inputs_each(self) -> None:
        for scenario in STANDARD_SCENARIOS:
            assert len(scenario.inputs) == 3, (
                f"Scenario {scenario.id!r} should have 3 inputs"
            )

    def test_standard_scenario_ids_are_unique(self) -> None:
        ids = [s.id for s in STANDARD_SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_standard_scenario_categories_covered(self) -> None:
        categories = {s.category for s in STANDARD_SCENARIOS}
        expected = {"classification", "extraction", "qa", "summarization", "multi_step"}
        assert categories == expected

    def test_standard_scenarios_have_expected_properties(self) -> None:
        for scenario in STANDARD_SCENARIOS:
            assert isinstance(scenario.expected_properties, dict)
            assert len(scenario.expected_properties) > 0


# ---------------------------------------------------------------------------
# ValidationHarness tests
# ---------------------------------------------------------------------------


class TestValidationHarness:
    def test_harness_creation(self) -> None:
        sim = _make_sim_env()
        real = _make_real_env()
        harness = ValidationHarness(sim_env=sim, real_env=real)
        assert harness.sim_environment is sim
        assert harness.real_environment is real

    def test_repr(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        assert "test-sim" in repr(harness)
        assert "test-real" in repr(harness)

    def test_run_scenario_matching_outputs_high_agreement(self) -> None:
        """When both environments return identical text, agreement should be ~1.0."""
        shared_responses = {
            "q1": "answer one",
            "q2": "answer two",
            "q3": "answer three",
        }
        sim = MockEnvironment(name="sim", responses=shared_responses)
        real = MockEnvironment(name="real", responses=shared_responses)
        harness = ValidationHarness(sim_env=sim, real_env=real)
        scenario = _make_scenario(["q1", "q2", "q3"])
        result = harness.run_scenario(scenario)
        assert result.agreement_score == pytest.approx(1.0, abs=1e-9)

    def test_run_scenario_returns_correct_output_counts(self) -> None:
        sim = _make_sim_env()
        real = _make_real_env()
        harness = ValidationHarness(sim_env=sim, real_env=real)
        scenario = _make_scenario(["p1", "p2", "p3"])
        result = harness.run_scenario(scenario)
        assert len(result.sim_outputs) == 3
        assert len(result.real_outputs) == 3

    def test_run_scenario_differing_outputs_lower_agreement(self) -> None:
        """When environments return very different text, agreement should be < 1.0."""
        sim = MockEnvironment(
            name="sim",
            responses={"q": "positive sentiment detected in the text"},
        )
        real = MockEnvironment(
            name="real",
            responses={"q": "xyz abc completely different content 123"},
        )
        harness = ValidationHarness(sim_env=sim, real_env=real)
        scenario = _make_scenario(["q"])
        result = harness.run_scenario(scenario)
        assert result.agreement_score < 1.0

    def test_run_scenario_differing_outputs_produces_result(self) -> None:
        sim = MockEnvironment(name="sim", responses={"q": "aaaa"})
        real = MockEnvironment(name="real", responses={"q": "zzzz"})
        harness = ValidationHarness(sim_env=sim, real_env=real)
        result = harness.run_scenario(_make_scenario(["q"]))
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == "test_scenario"

    def test_run_scenario_computes_latency_ratio(self) -> None:
        sim = MockEnvironment(name="sim", fixed_latency=0.010)
        real = MockEnvironment(name="real", fixed_latency=0.100)
        harness = ValidationHarness(sim_env=sim, real_env=real)
        scenario = _make_scenario(["p1"])
        result = harness.run_scenario(scenario)
        assert result.latency_ratio == pytest.approx(0.1, rel=1e-6)

    def test_run_all_uses_standard_scenarios_by_default(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        report = harness.run_all()
        assert len(report.scenarios) == len(STANDARD_SCENARIOS)

    def test_run_all_uses_provided_scenarios(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        custom = [
            _make_scenario(["p1", "p2"], scenario_id="custom_1"),
            _make_scenario(["p3"], scenario_id="custom_2"),
        ]
        report = harness.run_all(scenarios=custom)
        assert len(report.scenarios) == 2

    def test_run_all_sets_environment_names(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        report = harness.run_all(scenarios=[_make_scenario(["p"])])
        assert report.sim_environment == "test-sim"
        assert report.real_environment == "test-real"

    def test_run_all_five_standard_scenarios_produces_report(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        report = harness.run_all()
        assert isinstance(report, FidelityReport)
        assert len(report.scenarios) == 5

    def test_run_all_overall_fidelity_in_range(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        report = harness.run_all()
        assert 0.0 <= report.overall_fidelity <= 1.0


# ---------------------------------------------------------------------------
# FidelityReport tests
# ---------------------------------------------------------------------------


class TestFidelityReport:
    def _report_with_scores(
        self, agreement_scores: list[float], latency_ratios: list[float]
    ) -> FidelityReport:
        report = FidelityReport(sim_environment="sim", real_environment="real")
        for i, (ag, lr) in enumerate(zip(agreement_scores, latency_ratios)):
            result = ScenarioResult(
                scenario_id=f"s{i}",
                scenario_name=f"Scenario {i}",
                sim_outputs=[],
                real_outputs=[],
                agreement_score=ag,
                latency_ratio=lr,
            )
            report.scenarios.append(result)
        return report

    def test_overall_fidelity_empty(self) -> None:
        report = FidelityReport()
        assert report.overall_fidelity == 0.0

    def test_overall_fidelity_single_scenario(self) -> None:
        report = self._report_with_scores([0.8], [1.0])
        assert report.overall_fidelity == pytest.approx(0.8)

    def test_overall_fidelity_multiple_scenarios(self) -> None:
        report = self._report_with_scores([0.6, 0.8, 1.0], [1.0, 1.0, 1.0])
        expected = (0.6 + 0.8 + 1.0) / 3
        assert report.overall_fidelity == pytest.approx(expected)

    def test_overall_latency_ratio_empty(self) -> None:
        report = FidelityReport()
        assert report.overall_latency_ratio == 0.0

    def test_overall_latency_ratio_single_scenario(self) -> None:
        report = self._report_with_scores([1.0], [0.5])
        assert report.overall_latency_ratio == pytest.approx(0.5)

    def test_overall_latency_ratio_multiple_scenarios(self) -> None:
        report = self._report_with_scores([1.0, 1.0], [0.2, 0.4])
        expected = (0.2 + 0.4) / 2
        assert report.overall_latency_ratio == pytest.approx(expected)

    def test_to_dict_keys(self) -> None:
        report = self._report_with_scores([0.9], [0.5])
        data = report.to_dict()
        assert "timestamp" in data
        assert "sim_environment" in data
        assert "real_environment" in data
        assert "overall_fidelity" in data
        assert "overall_latency_ratio" in data
        assert "n_scenarios" in data
        assert "scenarios" in data

    def test_to_dict_n_scenarios(self) -> None:
        report = self._report_with_scores([0.9, 0.7, 0.8], [1.0, 1.0, 1.0])
        data = report.to_dict()
        assert data["n_scenarios"] == 3
        assert len(data["scenarios"]) == 3  # type: ignore[arg-type]

    def test_to_dict_overall_fidelity_matches_property(self) -> None:
        report = self._report_with_scores([0.75, 0.85], [1.0, 1.0])
        data = report.to_dict()
        assert data["overall_fidelity"] == pytest.approx(report.overall_fidelity)

    def test_to_markdown_returns_string(self) -> None:
        report = self._report_with_scores([0.9], [0.5])
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_to_markdown_contains_header(self) -> None:
        report = self._report_with_scores([0.9], [0.5])
        md = report.to_markdown()
        assert "Fidelity Report" in md

    def test_to_markdown_contains_environment_names(self) -> None:
        report = FidelityReport(sim_environment="my-sim", real_environment="my-real")
        md = report.to_markdown()
        assert "my-sim" in md
        assert "my-real" in md

    def test_to_markdown_empty_report(self) -> None:
        report = FidelityReport()
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "0" in md  # zero scenarios mentioned somewhere

    def test_to_markdown_verdict_pass(self) -> None:
        report = self._report_with_scores([0.9, 0.85], [1.0, 1.0])
        md = report.to_markdown()
        assert "PASS" in md

    def test_to_markdown_verdict_fail(self) -> None:
        report = self._report_with_scores([0.1, 0.2], [1.0, 1.0])
        md = report.to_markdown()
        assert "FAIL" in md

    def test_to_markdown_verdict_warn(self) -> None:
        report = self._report_with_scores([0.6, 0.65], [1.0, 1.0])
        md = report.to_markdown()
        assert "WARN" in md


# ---------------------------------------------------------------------------
# ScenarioResult tests
# ---------------------------------------------------------------------------


class TestScenarioResult:
    def test_creation_with_required_fields(self) -> None:
        result = ScenarioResult(
            scenario_id="s1",
            scenario_name="Test",
            sim_outputs=[],
            real_outputs=[],
            agreement_score=0.9,
            latency_ratio=0.5,
        )
        assert result.scenario_id == "s1"
        assert result.scenario_name == "Test"
        assert result.agreement_score == pytest.approx(0.9)
        assert result.latency_ratio == pytest.approx(0.5)
        assert result.details == {}

    def test_to_dict_keys(self) -> None:
        result = ScenarioResult(
            scenario_id="s1",
            scenario_name="Test",
            sim_outputs=[],
            real_outputs=[],
            agreement_score=0.8,
            latency_ratio=0.4,
        )
        data = result.to_dict()
        expected_keys = {
            "scenario_id",
            "scenario_name",
            "agreement_score",
            "latency_ratio",
            "n_sim_outputs",
            "n_real_outputs",
            "details",
        }
        assert expected_keys.issubset(data.keys())

    def test_to_dict_output_counts(self) -> None:
        out = EnvironmentOutput(response="r", latency_seconds=0.01)
        result = ScenarioResult(
            scenario_id="s1",
            scenario_name="Test",
            sim_outputs=[out, out],
            real_outputs=[out],
            agreement_score=0.5,
            latency_ratio=1.0,
        )
        data = result.to_dict()
        assert data["n_sim_outputs"] == 2
        assert data["n_real_outputs"] == 1


# ---------------------------------------------------------------------------
# Agreement computation tests
# ---------------------------------------------------------------------------


class TestAgreementComputation:
    def test_identical_texts_score_one(self) -> None:
        score = _text_similarity("hello world", "hello world")
        assert score == pytest.approx(1.0)

    def test_different_texts_score_less_than_one(self) -> None:
        score = _text_similarity("hello world", "completely unrelated xyz content")
        assert score < 1.0

    def test_completely_different_texts_score_low(self) -> None:
        score = _text_similarity("aaaa aaaa aaaa", "zzzz zzzz zzzz")
        assert score < 0.5

    def test_empty_harness_agreement_is_zero(self) -> None:
        harness = ValidationHarness(sim_env=_make_sim_env(), real_env=_make_real_env())
        score = harness._compute_agreement([], [])
        assert score == 0.0

    def test_agreement_is_case_insensitive(self) -> None:
        score_lower = _text_similarity("hello world", "hello world")
        score_mixed = _text_similarity("HELLO WORLD", "hello world")
        assert score_lower == pytest.approx(score_mixed)

    def test_latency_ratio_zero_real_latency(self) -> None:
        harness = ValidationHarness(sim_env=_make_sim_env(), real_env=_make_real_env())
        sim_out = [EnvironmentOutput(response="r", latency_seconds=0.01)]
        real_out = [EnvironmentOutput(response="r", latency_seconds=0.0)]
        ratio = harness._compute_latency_ratio(sim_out, real_out)
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# Environment ABC enforcement tests
# ---------------------------------------------------------------------------


class TestEnvironmentABC:
    def test_cannot_instantiate_abstract_environment(self) -> None:
        with pytest.raises(TypeError):
            Environment()  # type: ignore[abstract]

    def test_incomplete_subclass_raises_type_error(self) -> None:
        class IncompleteEnv(Environment):
            @property
            def name(self) -> str:
                return "incomplete"

            # Missing environment_type, execute, reset

        with pytest.raises(TypeError):
            IncompleteEnv()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class CompleteEnv(Environment):
            @property
            def name(self) -> str:
                return "complete"

            @property
            def environment_type(self) -> str:
                return "simulation"

            def execute(self, input_data: EnvironmentInput) -> EnvironmentOutput:
                return EnvironmentOutput(response="ok", latency_seconds=0.0)

            def reset(self) -> None:
                pass

        env = CompleteEnv()
        assert env.name == "complete"
        assert env.environment_type == "simulation"


# ---------------------------------------------------------------------------
# Integration tests: full pipeline sim -> real -> report
# ---------------------------------------------------------------------------


class TestIntegrationPipeline:
    def test_full_pipeline_identical_environments(self) -> None:
        """With identical environments, fidelity should be 1.0."""
        shared = {f"prompt_{i}": f"response_{i}" for i in range(15)}
        sim = MockEnvironment(name="sim", responses=shared)
        real = MockEnvironment(name="real", responses=shared)
        harness = ValidationHarness(sim_env=sim, real_env=real)
        report = harness.run_all()

        assert report.overall_fidelity == pytest.approx(1.0, abs=1e-9)
        assert len(report.scenarios) == 5
        assert report.sim_environment == "sim"
        assert report.real_environment == "real"

    def test_full_pipeline_divergent_environments(self) -> None:
        """With divergent responses, fidelity should be significantly below 1.0."""
        sim = MockEnvironment(name="sim", fixed_latency=0.001)
        real = MockEnvironment(name="real", fixed_latency=0.1)
        # Default fallback responses are different because the prompts contain
        # the environment-specific prefix so the text will differ structurally;
        # however both use the same fallback template, so we override for real:

        class DivergentEnv(Environment):
            @property
            def name(self) -> str:
                return "divergent"

            @property
            def environment_type(self) -> str:
                return "production"

            def execute(self, input_data: EnvironmentInput) -> EnvironmentOutput:
                return EnvironmentOutput(
                    response="zxqw 9876 completely unrelated output content",
                    latency_seconds=0.1,
                )

            def reset(self) -> None:
                pass

        harness = ValidationHarness(sim_env=sim, real_env=DivergentEnv())
        report = harness.run_all()
        assert report.overall_fidelity < 0.9

    def test_full_pipeline_report_serialises_to_dict(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        report = harness.run_all()
        data = report.to_dict()
        assert isinstance(data, dict)
        assert data["n_scenarios"] == 5
        assert isinstance(data["scenarios"], list)

    def test_full_pipeline_report_serialises_to_markdown(self) -> None:
        harness = ValidationHarness(
            sim_env=_make_sim_env(), real_env=_make_real_env()
        )
        report = harness.run_all()
        md = report.to_markdown()
        assert "test-sim" in md
        assert "test-real" in md
        # All five scenario names should appear
        for scenario in STANDARD_SCENARIOS:
            assert scenario.name in md

    def test_custom_single_scenario_pipeline(self) -> None:
        """End-to-end with one custom scenario and matching environments."""
        response = "The answer is Paris."
        sim = MockEnvironment(name="sim", responses={"q": response})
        real = MockEnvironment(name="real", responses={"q": response})
        harness = ValidationHarness(sim_env=sim, real_env=real)
        scenario = _make_scenario(["q"], scenario_id="custom_qa")
        report = harness.run_all(scenarios=[scenario])

        assert len(report.scenarios) == 1
        assert report.scenarios[0].agreement_score == pytest.approx(1.0, abs=1e-9)
        assert report.overall_fidelity == pytest.approx(1.0, abs=1e-9)
