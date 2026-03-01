#!/usr/bin/env python3
"""Example: Skill Library and Composition

Demonstrates defining reusable agent skills, composing them into
composite skills, and managing them in a skill library.

Usage:
    python examples/06_skill_library.py

Requirements:
    pip install agent-sim-bridge
"""
from __future__ import annotations

import agent_sim_bridge
from agent_sim_bridge import (
    CompositeSkill,
    Skill,
    SkillComposer,
    SkillLibrary,
    SkillNotFoundError,
    SkillResult,
    SkillStatus,
)


def make_skill(name: str, description: str) -> Skill:
    """Create a simple skill with a callable."""
    def execute(context: dict[str, object]) -> SkillResult:
        return SkillResult(
            skill_name=name,
            status=SkillStatus.SUCCESS,
            output={name: f"completed with context keys={list(context.keys())}"},
        )
    return Skill(name=name, description=description, execute=execute)


def main() -> None:
    print(f"agent-sim-bridge version: {agent_sim_bridge.__version__}")

    # Step 1: Define individual skills
    navigate = make_skill("navigate", "Navigate to a target position.")
    pick = make_skill("pick", "Pick up an object.")
    place = make_skill("place", "Place an object at a target location.")
    inspect = make_skill("inspect", "Inspect an object for quality.")

    # Step 2: Build a skill library
    library = SkillLibrary()
    for skill in [navigate, pick, place, inspect]:
        library.register(skill)
    print(f"Skill library: {library.count()} skills registered")

    # Retrieve a skill
    try:
        retrieved = library.get("navigate")
        result = retrieved.execute({"target": "shelf-A3"})
        print(f"  navigate: {result.status.value} — {result.output}")
    except SkillNotFoundError as error:
        print(f"  Skill not found: {error}")

    # Step 3: Compose skills into a workflow
    composer = SkillComposer()
    pick_and_place: CompositeSkill = composer.compose(
        name="pick-and-place",
        skills=[navigate, pick, navigate, place],
        description="Navigate, pick, navigate to target, and place.",
    )
    context = {"source": "bin-1", "target": "shelf-B2", "object": "widget-7"}
    composite_result = pick_and_place.execute(context)
    print(f"\nComposite skill '{pick_and_place.name}':")
    print(f"  Status: {composite_result.status.value}")
    for step_name, step_out in composite_result.step_outputs.items():
        print(f"  [{step_name}] {step_out}")

    # Step 4: List all skills
    all_skills = library.list()
    print(f"\nAll skills ({len(all_skills)}):")
    for skill in all_skills:
        print(f"  {skill.name}: {skill.description}")


if __name__ == "__main__":
    main()
