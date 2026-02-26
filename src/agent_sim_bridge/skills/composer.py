"""SkillComposer — compose multiple skills into a sequential composite skill.

The composer chains skills so they execute one after another in the same
environment without resetting between them.  If any skill fails, the composite
fails immediately (fail-fast semantics) unless ``continue_on_failure=True``
is set.

CompositeSkill
--------------
:class:`CompositeSkill` is itself a :class:`~agent_sim_bridge.skills.base.Skill`,
so composites can be nested and treated uniformly by the
:class:`~agent_sim_bridge.skills.library.SkillLibrary`.
"""
from __future__ import annotations

import logging

from agent_sim_bridge.skills.base import Skill, SkillResult, SkillStatus

logger = logging.getLogger(__name__)


class CompositeSkill(Skill):
    """A :class:`~agent_sim_bridge.skills.base.Skill` that runs sub-skills sequentially.

    Parameters
    ----------
    name:
        Name of this composite skill.
    skills:
        Ordered list of sub-skills to execute.
    continue_on_failure:
        When False (default) execution stops at the first failing sub-skill.
        When True all sub-skills are attempted regardless of individual results.
    max_steps:
        Combined step budget across all sub-skills.  ``None`` means no limit
        beyond the environment's own.
    """

    def __init__(
        self,
        name: str,
        skills: list[Skill],
        continue_on_failure: bool = False,
        max_steps: int | None = None,
    ) -> None:
        super().__init__(name=name, max_steps=max_steps)
        if not skills:
            raise ValueError(f"CompositeSkill {name!r} must contain at least one sub-skill.")
        self._skills = list(skills)
        self._continue_on_failure = continue_on_failure

    @property
    def skills(self) -> list[Skill]:
        """The ordered list of sub-skills."""
        return list(self._skills)

    def execute(
        self,
        env: object,
        context: dict[str, object] | None = None,
    ) -> SkillResult:
        """Execute all sub-skills sequentially.

        Parameters
        ----------
        env:
            The environment shared across all sub-skills.
        context:
            Optional context passed to each sub-skill.  Sub-skill results
            are merged into a copy of the context so later skills can read
            earlier results under the key ``"<skill_name>_result"``.

        Returns
        -------
        SkillResult
            Aggregated result with cumulative reward and step count.
        """
        running_context: dict[str, object] = dict(context or {})
        total_reward = 0.0
        total_steps = 0
        sub_results: list[SkillResult] = []
        final_status = SkillStatus.SUCCESS
        error_message: str | None = None

        for skill in self._skills:
            # Enforce composite step budget.
            if self._max_steps is not None and total_steps >= self._max_steps:
                logger.warning(
                    "CompositeSkill %r step budget (%d) exhausted after %d steps; "
                    "remaining sub-skills skipped.",
                    self._name,
                    self._max_steps,
                    total_steps,
                )
                final_status = SkillStatus.TIMEOUT
                break

            logger.debug(
                "CompositeSkill %r executing sub-skill %r (step %d/%s).",
                self._name,
                skill.name,
                total_steps,
                str(self._max_steps) if self._max_steps else "unlimited",
            )

            result = skill.execute(env, context=dict(running_context))
            sub_results.append(result)
            total_reward += result.total_reward
            total_steps += result.steps_taken

            # Propagate sub-skill result into context for downstream skills.
            running_context[f"{skill.name}_result"] = result.summary()

            if not result.succeeded:
                logger.warning(
                    "Sub-skill %r failed (%s); continuing=%s.",
                    skill.name,
                    result.status.value,
                    self._continue_on_failure,
                )
                if final_status == SkillStatus.SUCCESS:
                    # Record first failure.
                    final_status = result.status
                    error_message = result.error
                if not self._continue_on_failure:
                    break

        return SkillResult(
            skill_name=self._name,
            status=final_status,
            total_reward=total_reward,
            steps_taken=total_steps,
            data={
                "sub_results": [r.summary() for r in sub_results],
                "n_sub_skills": len(self._skills),
                "n_executed": len(sub_results),
            },
            error=error_message,
        )

    def __repr__(self) -> str:
        return (
            f"CompositeSkill(name={self._name!r}, "
            f"n_skills={len(self._skills)}, "
            f"continue_on_failure={self._continue_on_failure})"
        )


class SkillComposer:
    """Factory that creates :class:`CompositeSkill` instances.

    Using this class rather than constructing :class:`CompositeSkill` directly
    makes the intent explicit in calling code and allows additional validation
    logic to be added in one place.

    Example
    -------
    ::

        composer = SkillComposer()
        composite = composer.compose(
            "pick_and_place",
            skills=[reach_skill, grasp_skill, move_skill, release_skill],
        )
    """

    def compose(
        self,
        name: str,
        skills: list[Skill],
        continue_on_failure: bool = False,
        max_steps: int | None = None,
    ) -> CompositeSkill:
        """Create a :class:`CompositeSkill` from a list of sub-skills.

        Parameters
        ----------
        name:
            Name for the new composite skill.
        skills:
            Ordered sub-skills to chain.
        continue_on_failure:
            When True, all sub-skills run regardless of individual failures.
        max_steps:
            Combined step budget.

        Returns
        -------
        CompositeSkill
        """
        composite = CompositeSkill(
            name=name,
            skills=skills,
            continue_on_failure=continue_on_failure,
            max_steps=max_steps,
        )
        logger.debug(
            "Composed skill %r from %d sub-skills.",
            name,
            len(skills),
        )
        return composite
