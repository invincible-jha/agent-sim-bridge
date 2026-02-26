"""ScenarioManager — define, save, and load test scenarios.

A *scenario* is a named, reproducible configuration that can be used to
initialise a simulation episode: initial state overrides, environment
parameter overrides, policy hints, and expected outcome criteria.

Scenarios are persisted as YAML files for human readability.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScenarioOutcomeCriteria(BaseModel):
    """Success/failure criteria for a scenario.

    Attributes
    ----------
    min_reward:
        Episode must achieve at least this total reward to be considered
        successful.  ``None`` means no reward threshold.
    max_steps:
        Episode must complete within this many steps.  ``None`` means no
        limit beyond the environment's own.
    require_termination:
        If True, the episode must end with ``terminated=True`` (not just
        truncated) to count as a success.
    """

    min_reward: float | None = None
    max_steps: int | None = None
    require_termination: bool = False


class Scenario(BaseModel):
    """A named, serialisable simulation scenario.

    Attributes
    ----------
    name:
        Unique scenario identifier.
    description:
        Human-readable description.
    seed:
        RNG seed for reproducibility.
    initial_state:
        Optional mapping of state variable names to override values.
    env_parameters:
        Key/value pairs forwarded to the environment's ``reset`` options.
    max_episode_steps:
        Override the environment's own step limit for this scenario.
        ``None`` inherits from the environment.
    tags:
        Free-form labels for filtering/grouping.
    outcome:
        Optional success/failure criteria.
    metadata:
        Arbitrary extra fields.
    """

    name: str
    description: str = ""
    seed: int | None = None
    initial_state: dict[str, float] = Field(default_factory=dict)
    env_parameters: dict[str, object] = Field(default_factory=dict)
    max_episode_steps: int | None = None
    tags: list[str] = Field(default_factory=list)
    outcome: ScenarioOutcomeCriteria = Field(default_factory=ScenarioOutcomeCriteria)
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_reset_options(self) -> dict[str, object]:
        """Merge initial_state and env_parameters into a reset-options dict."""
        options: dict[str, object] = {}
        if self.initial_state:
            options["initial_state"] = dict(self.initial_state)
        options.update(self.env_parameters)
        return options

    def evaluate(self, total_reward: float, steps: int, terminated: bool) -> bool:
        """Return True if the episode outcome satisfies this scenario's criteria.

        Parameters
        ----------
        total_reward:
            Cumulative reward achieved in the episode.
        steps:
            Number of steps taken.
        terminated:
            Whether the episode ended in a terminal state.
        """
        if self.outcome.min_reward is not None and total_reward < self.outcome.min_reward:
            return False
        if self.outcome.max_steps is not None and steps > self.outcome.max_steps:
            return False
        if self.outcome.require_termination and not terminated:
            return False
        return True


class ScenarioManager:
    """Persist and retrieve :class:`Scenario` objects on disk.

    Each scenario is stored as a separate YAML file inside a directory.
    The file name is derived from the scenario ``name`` by replacing spaces
    with underscores and lowercasing.

    Parameters
    ----------
    directory:
        Directory where scenario files are stored.  Created on first save
        if it does not exist.
    """

    def __init__(self, directory: str | Path = "scenarios") -> None:
        self._directory = Path(directory)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, scenario: Scenario) -> Path:
        """Write *scenario* to ``<directory>/<name>.yaml``.

        Parameters
        ----------
        scenario:
            The scenario to persist.

        Returns
        -------
        Path
            The file path the scenario was written to.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        filename = self._name_to_filename(scenario.name)
        path = self._directory / filename
        data = scenario.model_dump()
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info("Saved scenario %r to %s", scenario.name, path)
        return path

    def load(self, name: str) -> Scenario:
        """Load a scenario by name.

        Parameters
        ----------
        name:
            The scenario's ``name`` field (not the file name).

        Returns
        -------
        Scenario

        Raises
        ------
        FileNotFoundError
            If no file exists for the given name.
        """
        filename = self._name_to_filename(name)
        path = self._directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Scenario {name!r} not found at {path}. "
                f"Available scenarios: {self.list_names()}"
            )
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        scenario = Scenario.model_validate(data)
        logger.debug("Loaded scenario %r from %s", name, path)
        return scenario

    def delete(self, name: str) -> None:
        """Remove a scenario file.

        Parameters
        ----------
        name:
            Scenario name.

        Raises
        ------
        FileNotFoundError
            If the scenario does not exist.
        """
        filename = self._name_to_filename(name)
        path = self._directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Scenario {name!r} not found at {path}.")
        path.unlink()
        logger.info("Deleted scenario %r (%s)", name, path)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_names(self) -> list[str]:
        """Return sorted list of all scenario names on disk."""
        if not self._directory.exists():
            return []
        names: list[str] = []
        for yaml_path in sorted(self._directory.glob("*.yaml")):
            try:
                with yaml_path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                names.append(str(data.get("name", yaml_path.stem)))
            except Exception:  # noqa: BLE001
                logger.warning("Could not parse scenario file %s", yaml_path)
        return names

    def list_all(self) -> list[Scenario]:
        """Load and return all scenarios."""
        return [self.load(name) for name in self.list_names()]

    def filter_by_tag(self, tag: str) -> list[Scenario]:
        """Return all scenarios that include *tag* in their tag list."""
        return [s for s in self.list_all() if tag in s.tags]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _name_to_filename(name: str) -> str:
        safe = name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
        return f"{safe}.yaml"

    def __repr__(self) -> str:
        return f"ScenarioManager(directory={str(self._directory)!r})"
