"""Unit tests for skills modules.

Covers:
- skills/base.py: SkillStatus, SkillResult, Skill ABC (name, max_steps, repr)
- skills/composer.py: CompositeSkill (sequential execution, fail-fast,
  continue_on_failure, step budget timeout, context propagation),
  SkillComposer.compose
- skills/library.py: SkillLibrary (register, register_or_replace, remove,
  get, search, list_all, list_names, __contains__, __len__, __repr__),
  SkillNotFoundError
"""
from __future__ import annotations

import pytest

from agent_sim_bridge.skills.base import Skill, SkillResult, SkillStatus
from agent_sim_bridge.skills.composer import CompositeSkill, SkillComposer
from agent_sim_bridge.skills.library import SkillLibrary, SkillNotFoundError


# ---------------------------------------------------------------------------
# Minimal concrete Skill implementation
# ---------------------------------------------------------------------------


class FixedResultSkill(Skill):
    """A skill that returns a pre-configured result."""

    def __init__(
        self,
        name: str,
        status: SkillStatus = SkillStatus.SUCCESS,
        reward: float = 1.0,
        steps: int = 1,
        max_steps: int | None = None,
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, max_steps=max_steps)
        self._status = status
        self._reward = reward
        self._steps = steps
        self.tags: list[str] = tags or []

    def execute(
        self,
        env: object,
        context: dict[str, object] | None = None,
    ) -> SkillResult:
        return SkillResult(
            skill_name=self._name,
            status=self._status,
            total_reward=self._reward,
            steps_taken=self._steps,
            error="fail" if self._status != SkillStatus.SUCCESS else None,
        )


# ---------------------------------------------------------------------------
# SkillStatus & SkillResult
# ---------------------------------------------------------------------------


class TestSkillResult:
    def test_succeeded_true_for_success(self) -> None:
        result = SkillResult(skill_name="s", status=SkillStatus.SUCCESS)
        assert result.succeeded is True

    def test_succeeded_false_for_failure(self) -> None:
        result = SkillResult(skill_name="s", status=SkillStatus.FAILURE)
        assert result.succeeded is False

    def test_succeeded_false_for_aborted(self) -> None:
        result = SkillResult(skill_name="s", status=SkillStatus.ABORTED)
        assert result.succeeded is False

    def test_summary_keys(self) -> None:
        result = SkillResult(skill_name="reach", status=SkillStatus.SUCCESS, total_reward=5.0)
        summary = result.summary()
        assert "skill_name" in summary
        assert "status" in summary
        assert "succeeded" in summary
        assert "total_reward" in summary

    def test_summary_values(self) -> None:
        result = SkillResult(
            skill_name="pick",
            status=SkillStatus.TIMEOUT,
            total_reward=2.5,
            steps_taken=10,
        )
        s = result.summary()
        assert s["skill_name"] == "pick"
        assert s["status"] == "timeout"
        assert s["total_reward"] == 2.5


# ---------------------------------------------------------------------------
# Skill ABC
# ---------------------------------------------------------------------------


class TestSkillABC:
    def test_name_property(self) -> None:
        skill = FixedResultSkill("my-skill")
        assert skill.name == "my-skill"

    def test_max_steps_property_none(self) -> None:
        skill = FixedResultSkill("s")
        assert skill.max_steps is None

    def test_max_steps_property_set(self) -> None:
        skill = FixedResultSkill("s", max_steps=20)
        assert skill.max_steps == 20

    def test_repr_contains_name(self) -> None:
        skill = FixedResultSkill("navigate")
        assert "navigate" in repr(skill)

    def test_execute_returns_skill_result(self) -> None:
        skill = FixedResultSkill("s")
        result = skill.execute(env=None)
        assert isinstance(result, SkillResult)


# ---------------------------------------------------------------------------
# CompositeSkill
# ---------------------------------------------------------------------------


class TestCompositeSkill:
    def test_empty_skills_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one sub-skill"):
            CompositeSkill("composite", skills=[])

    def test_skills_property(self) -> None:
        s1 = FixedResultSkill("a")
        s2 = FixedResultSkill("b")
        composite = CompositeSkill("comp", skills=[s1, s2])
        assert composite.skills == [s1, s2]

    def test_execute_all_success(self) -> None:
        s1 = FixedResultSkill("a", reward=2.0, steps=3)
        s2 = FixedResultSkill("b", reward=3.0, steps=2)
        composite = CompositeSkill("comp", skills=[s1, s2])
        result = composite.execute(env=None)
        assert result.status == SkillStatus.SUCCESS
        assert result.total_reward == pytest.approx(5.0)
        assert result.steps_taken == 5

    def test_execute_fail_fast(self) -> None:
        s1 = FixedResultSkill("a", status=SkillStatus.FAILURE)
        s2 = FixedResultSkill("b", reward=10.0)
        composite = CompositeSkill("comp", skills=[s1, s2], continue_on_failure=False)
        result = composite.execute(env=None)
        assert result.status == SkillStatus.FAILURE
        # s2 should not have run — reward is from s1 only.
        assert result.total_reward == pytest.approx(1.0)  # s1's default reward
        assert result.data["n_executed"] == 1

    def test_execute_continue_on_failure(self) -> None:
        s1 = FixedResultSkill("a", status=SkillStatus.FAILURE, reward=1.0, steps=1)
        s2 = FixedResultSkill("b", status=SkillStatus.SUCCESS, reward=5.0, steps=2)
        composite = CompositeSkill("comp", skills=[s1, s2], continue_on_failure=True)
        result = composite.execute(env=None)
        # Both run; overall status should be failure (first failure wins).
        assert result.status == SkillStatus.FAILURE
        assert result.total_reward == pytest.approx(6.0)
        assert result.data["n_executed"] == 2

    def test_step_budget_stops_execution(self) -> None:
        s1 = FixedResultSkill("a", steps=3)
        s2 = FixedResultSkill("b", steps=3)
        # Budget is 3, so s2 should not execute.
        composite = CompositeSkill("comp", skills=[s1, s2], max_steps=3)
        result = composite.execute(env=None)
        assert result.status == SkillStatus.TIMEOUT
        assert result.data["n_executed"] == 1

    def test_context_propagated_to_sub_skills(self) -> None:
        received_contexts: list[dict[str, object]] = []

        class ContextCapture(Skill):
            def execute(
                self, env: object, context: dict[str, object] | None = None
            ) -> SkillResult:
                received_contexts.append(dict(context or {}))
                return SkillResult(skill_name=self._name, status=SkillStatus.SUCCESS)

        s1 = ContextCapture("first")
        s2 = ContextCapture("second")
        composite = CompositeSkill("comp", skills=[s1, s2])
        composite.execute(env=None, context={"initial": 1})
        # Second call should contain "first_result" injected by composite.
        assert "first_result" in received_contexts[1]

    def test_repr_contains_name_and_n_skills(self) -> None:
        s1 = FixedResultSkill("x")
        composite = CompositeSkill("my-composite", skills=[s1])
        result = repr(composite)
        assert "my-composite" in result
        assert "1" in result

    def test_sub_results_in_data(self) -> None:
        s1 = FixedResultSkill("a")
        s2 = FixedResultSkill("b")
        composite = CompositeSkill("comp", skills=[s1, s2])
        result = composite.execute(env=None)
        sub_results = result.data["sub_results"]
        assert isinstance(sub_results, list)
        assert len(sub_results) == 2


# ---------------------------------------------------------------------------
# SkillComposer
# ---------------------------------------------------------------------------


class TestSkillComposer:
    def test_compose_returns_composite_skill(self) -> None:
        composer = SkillComposer()
        s1 = FixedResultSkill("alpha")
        composite = composer.compose("my-composite", skills=[s1])
        assert isinstance(composite, CompositeSkill)
        assert composite.name == "my-composite"

    def test_compose_with_options(self) -> None:
        composer = SkillComposer()
        s1 = FixedResultSkill("a")
        composite = composer.compose(
            "combo",
            skills=[s1],
            continue_on_failure=True,
            max_steps=50,
        )
        assert composite.max_steps == 50

    def test_compose_empty_skills_raises(self) -> None:
        composer = SkillComposer()
        with pytest.raises(ValueError):
            composer.compose("empty", skills=[])


# ---------------------------------------------------------------------------
# SkillLibrary
# ---------------------------------------------------------------------------


class TestSkillLibrary:
    def setup_method(self) -> None:
        self.library = SkillLibrary()

    def _skill(self, name: str, tags: list[str] | None = None) -> FixedResultSkill:
        return FixedResultSkill(name, tags=tags)

    def test_register_and_get(self) -> None:
        skill = self._skill("reach")
        self.library.register(skill)
        retrieved = self.library.get("reach")
        assert retrieved is skill

    def test_register_duplicate_raises(self) -> None:
        self.library.register(self._skill("reach"))
        with pytest.raises(ValueError, match="already in the library"):
            self.library.register(self._skill("reach"))

    def test_register_or_replace_no_existing(self) -> None:
        skill = self._skill("grasp")
        self.library.register_or_replace(skill)
        assert "grasp" in self.library

    def test_register_or_replace_replaces_existing(self) -> None:
        old_skill = self._skill("grasp")
        new_skill = FixedResultSkill("grasp", reward=999.0)
        self.library.register(old_skill)
        self.library.register_or_replace(new_skill)
        assert self.library.get("grasp") is new_skill

    def test_remove_existing(self) -> None:
        self.library.register(self._skill("wave"))
        self.library.remove("wave")
        assert "wave" not in self.library

    def test_remove_not_found_raises(self) -> None:
        with pytest.raises(SkillNotFoundError):
            self.library.remove("ghost")

    def test_get_not_found_raises(self) -> None:
        with pytest.raises(SkillNotFoundError):
            self.library.get("phantom")

    def test_skill_not_found_error_has_name(self) -> None:
        error = SkillNotFoundError("missing-skill")
        assert error.skill_name == "missing-skill"
        assert "missing-skill" in str(error)

    def test_search_by_name_substring(self) -> None:
        self.library.register(self._skill("reach-forward"))
        self.library.register(self._skill("grasp-object"))
        results = self.library.search("reach")
        assert len(results) == 1
        assert results[0].name == "reach-forward"

    def test_search_empty_query_returns_all(self) -> None:
        self.library.register(self._skill("a"))
        self.library.register(self._skill("b"))
        assert len(self.library.search("")) == 2

    def test_search_by_tag(self) -> None:
        self.library.register(self._skill("navigate", tags=["mobility"]))
        self.library.register(self._skill("pick", tags=["manipulation"]))
        results = self.library.search("mobility")
        assert len(results) == 1
        assert results[0].name == "navigate"

    def test_search_case_insensitive(self) -> None:
        self.library.register(self._skill("UPPERCASE"))
        results = self.library.search("upper")
        assert len(results) == 1

    def test_search_returns_sorted(self) -> None:
        self.library.register(self._skill("z-skill"))
        self.library.register(self._skill("a-skill"))
        results = self.library.search("")
        assert results[0].name == "a-skill"
        assert results[1].name == "z-skill"

    def test_list_all_sorted(self) -> None:
        self.library.register(self._skill("z"))
        self.library.register(self._skill("a"))
        names = [s.name for s in self.library.list_all()]
        assert names == ["a", "z"]

    def test_list_names_sorted(self) -> None:
        self.library.register(self._skill("c"))
        self.library.register(self._skill("a"))
        assert self.library.list_names() == ["a", "c"]

    def test_contains(self) -> None:
        self.library.register(self._skill("exists"))
        assert "exists" in self.library
        assert "missing" not in self.library

    def test_len(self) -> None:
        assert len(self.library) == 0
        self.library.register(self._skill("one"))
        assert len(self.library) == 1

    def test_repr(self) -> None:
        self.library.register(self._skill("skill-a"))
        result = repr(self.library)
        assert "SkillLibrary" in result
        assert "skill-a" in result
