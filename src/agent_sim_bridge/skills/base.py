"""Skill ABC and SkillResult dataclass.

A *skill* is a reusable, named behaviour that can be executed within an
environment.  Skills compose into higher-level behaviours via
:class:`~agent_sim_bridge.skills.composer.SkillComposer`.

Design principles
-----------------
* Skills are stateless between executions — all episode-specific data is
  passed through ``context`` and returned in :class:`SkillResult`.
* Skills do not own environments — they receive one at execution time.
* The :class:`SkillResult` carries both the outcome and structured data so
  callers can branch on success/failure without parsing log output.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class SkillStatus(str, Enum):
    """Terminal status of a skill execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"
    TIMEOUT = "timeout"


@dataclass
class SkillResult:
    """Outcome of one :class:`Skill` execution.

    Attributes
    ----------
    skill_name:
        Name of the skill that produced this result.
    status:
        Terminal status.
    total_reward:
        Cumulative reward accumulated while the skill was running.
    steps_taken:
        Number of environment steps the skill consumed.
    data:
        Arbitrary structured output from the skill (e.g., a detected
        object pose, a computed trajectory, etc.).
    error:
        Exception message if the skill aborted unexpectedly.  ``None``
        when :attr:`status` is SUCCESS.
    """

    skill_name: str
    status: SkillStatus = SkillStatus.SUCCESS
    total_reward: float = 0.0
    steps_taken: int = 0
    data: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when :attr:`status` is SUCCESS."""
        return self.status == SkillStatus.SUCCESS

    def summary(self) -> dict[str, object]:
        """Return a plain-dict summary."""
        return {
            "skill_name": self.skill_name,
            "status": self.status.value,
            "total_reward": self.total_reward,
            "steps_taken": self.steps_taken,
            "succeeded": self.succeeded,
            "error": self.error,
        }


class Skill(ABC):
    """Abstract base class for reusable agent skills.

    Subclasses must implement :meth:`execute`.

    Parameters
    ----------
    name:
        Human-readable skill identifier.  Used in logging and :class:`SkillResult`.
    max_steps:
        Hard cap on environment steps.  Pass ``None`` to inherit from the
        environment's own limit.
    """

    def __init__(self, name: str, max_steps: int | None = None) -> None:
        self._name = name
        self._max_steps = max_steps

    @property
    def name(self) -> str:
        """Skill identifier."""
        return self._name

    @property
    def max_steps(self) -> int | None:
        """Step budget for this skill, or ``None`` for unlimited."""
        return self._max_steps

    @abstractmethod
    def execute(
        self,
        env: object,
        context: dict[str, object] | None = None,
    ) -> SkillResult:
        """Run the skill in ``env``.

        Parameters
        ----------
        env:
            The environment to act in.  The skill calls ``env.step(action)``
            internally and must not call ``env.reset()``.
        context:
            Optional key/value context from the caller (e.g., goal position,
            prior skill results).

        Returns
        -------
        SkillResult
            The execution outcome.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, max_steps={self._max_steps!r})"
