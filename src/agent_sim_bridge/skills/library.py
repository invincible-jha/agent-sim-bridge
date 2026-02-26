"""SkillLibrary — registry for reusable agent skills.

The library stores :class:`~agent_sim_bridge.skills.base.Skill` *instances*
(not classes), so all configuration is baked in at registration time.  This
makes retrieval trivially fast and keeps the caller's code clean.

Search is performed by matching the query string against skill names and tags
(if the skill has a ``tags`` attribute).
"""
from __future__ import annotations

import logging

from agent_sim_bridge.skills.base import Skill

logger = logging.getLogger(__name__)


class SkillNotFoundError(KeyError):
    """Raised when a requested skill is not in the library."""

    def __init__(self, name: str) -> None:
        self.skill_name = name
        super().__init__(f"Skill {name!r} is not in the library.")


class SkillLibrary:
    """In-memory registry of :class:`~agent_sim_bridge.skills.base.Skill` instances.

    Skills are stored and retrieved by name.  The library supports fuzzy
    search by substring matching against both the skill name and an optional
    ``tags`` attribute.

    Example
    -------
    ::

        library = SkillLibrary()
        library.register(ReachSkill("reach", max_steps=50))
        library.register(GraspSkill("grasp", max_steps=20))

        reach = library.get("reach")
        manipulation_skills = library.search("grasp")
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Add a skill to the library.

        Parameters
        ----------
        skill:
            The :class:`~agent_sim_bridge.skills.base.Skill` instance to register.

        Raises
        ------
        ValueError
            If a skill with the same name is already registered.
        """
        if skill.name in self._skills:
            raise ValueError(
                f"Skill {skill.name!r} is already in the library. "
                "Use a unique name or remove the existing skill first."
            )
        self._skills[skill.name] = skill
        logger.debug("Registered skill %r (%s).", skill.name, type(skill).__name__)

    def register_or_replace(self, skill: Skill) -> None:
        """Register a skill, silently replacing any existing skill with the same name.

        Parameters
        ----------
        skill:
            The :class:`~agent_sim_bridge.skills.base.Skill` instance to register.
        """
        replaced = skill.name in self._skills
        self._skills[skill.name] = skill
        if replaced:
            logger.debug("Replaced existing skill %r.", skill.name)
        else:
            logger.debug("Registered skill %r (%s).", skill.name, type(skill).__name__)

    def remove(self, name: str) -> None:
        """Remove a skill from the library by name.

        Parameters
        ----------
        name:
            The skill name.

        Raises
        ------
        SkillNotFoundError
            If no skill with that name exists.
        """
        if name not in self._skills:
            raise SkillNotFoundError(name)
        del self._skills[name]
        logger.debug("Removed skill %r from library.", name)

    def get(self, name: str) -> Skill:
        """Retrieve a skill by exact name.

        Parameters
        ----------
        name:
            The skill name used at registration.

        Returns
        -------
        Skill

        Raises
        ------
        SkillNotFoundError
            If no skill with that name exists.
        """
        try:
            return self._skills[name]
        except KeyError:
            raise SkillNotFoundError(name) from None

    def search(self, query: str) -> list[Skill]:
        """Find skills whose name or tags contain ``query`` (case-insensitive).

        Parameters
        ----------
        query:
            Substring to search for.  An empty string returns all skills.

        Returns
        -------
        list[Skill]
            Matching skills, sorted alphabetically by name.
        """
        lower_query = query.lower()
        results: list[Skill] = []
        for skill in self._skills.values():
            if lower_query in skill.name.lower():
                results.append(skill)
                continue
            # Check tags if the skill exposes them.
            tags: list[str] = getattr(skill, "tags", [])
            if any(lower_query in tag.lower() for tag in tags):
                results.append(skill)
        results.sort(key=lambda s: s.name)
        return results

    def list_all(self) -> list[Skill]:
        """Return all registered skills sorted alphabetically by name.

        Returns
        -------
        list[Skill]
        """
        return sorted(self._skills.values(), key=lambda s: s.name)

    def list_names(self) -> list[str]:
        """Return sorted list of all registered skill names."""
        return sorted(self._skills)

    def __contains__(self, name: object) -> bool:
        """Support ``"my-skill" in library`` membership test."""
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillLibrary(n_skills={len(self._skills)}, names={self.list_names()})"
