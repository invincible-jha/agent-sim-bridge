"""SafetyConstraint and SafetyChecker — declarative action validation.

Safety constraints define admissibility conditions on actions.  The checker
evaluates an action against a list of constraints and returns all violations
found, so callers can decide how to respond (warn, block, emergency stop).

Design principles
-----------------
* Constraints are pure data (dataclasses) — no logic embedded in them.
* The checker is stateless: it can be called from any thread.
* All violations are collected before returning (fail-all, not fail-fast),
  giving operators a complete picture of what went wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Supported constraint check modes."""

    RANGE = "range"
    MAX_RATE = "max_rate"
    FORBIDDEN_ZONE = "forbidden_zone"
    CUSTOM = "custom"


class ViolationSeverity(str, Enum):
    """How serious a constraint violation is."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SafetyConstraint:
    """Specifies one safety requirement on an action vector.

    Attributes
    ----------
    name:
        Human-readable constraint identifier.
    constraint_type:
        The kind of check to perform.
    dimension:
        Which action dimension this constraint applies to.  Use ``-1``
        to indicate the constraint applies to the L2 norm of the full
        action vector.
    min_value:
        Inclusive lower bound (used by RANGE and FORBIDDEN_ZONE).
    max_value:
        Inclusive upper bound (used by RANGE and FORBIDDEN_ZONE).
        For FORBIDDEN_ZONE the action must *not* fall in ``[min_value, max_value]``.
    max_delta:
        Maximum allowed change from the previous action (used by MAX_RATE).
    severity:
        How serious a violation of this constraint is.
    description:
        Optional explanatory text.
    """

    name: str
    constraint_type: ConstraintType = ConstraintType.RANGE
    dimension: int = 0
    min_value: float = float("-inf")
    max_value: float = float("inf")
    max_delta: float = float("inf")
    severity: ViolationSeverity = ViolationSeverity.ERROR
    description: str = ""


@dataclass
class SafetyViolation:
    """Records one constraint violation detected by :class:`SafetyChecker`.

    Attributes
    ----------
    constraint_name:
        Name of the violated :class:`SafetyConstraint`.
    severity:
        Copied from the constraint.
    actual_value:
        The action value (or derived quantity) that triggered the violation.
    message:
        Human-readable description of the violation.
    dimension:
        Which dimension was violated, or ``-1`` for norm-based checks.
    """

    constraint_name: str
    severity: ViolationSeverity
    actual_value: float
    message: str
    dimension: int = 0


class SafetyChecker:
    """Evaluate a list of constraints against an action vector.

    The checker is intentionally stateless with respect to constraint
    registration — pass the constraint list at each call so it can be
    updated externally without reconstructing the checker.

    Parameters
    ----------
    previous_action:
        The last action sent to the environment, used for MAX_RATE checks.
        Update it after each successful step via :meth:`update_previous_action`.

    Example
    -------
    ::

        constraints = [
            SafetyConstraint("joint_0_range", dimension=0, min_value=-1.0, max_value=1.0),
            SafetyConstraint("joint_1_range", dimension=1, min_value=-1.0, max_value=1.0),
        ]
        checker = SafetyChecker()
        violations = checker.check([0.5, 2.0], constraints)
        # [SafetyViolation(constraint_name="joint_1_range", ...)]
    """

    def __init__(self, previous_action: list[float] | None = None) -> None:
        self._previous_action: list[float] | None = previous_action

    def update_previous_action(self, action: list[float]) -> None:
        """Update the stored previous action for MAX_RATE checks.

        Parameters
        ----------
        action:
            The action that was just applied to the environment.
        """
        self._previous_action = list(action)

    def check(
        self,
        action: list[float],
        constraints: list[SafetyConstraint],
    ) -> list[SafetyViolation]:
        """Check *action* against every constraint and return all violations.

        Parameters
        ----------
        action:
            The action vector to validate.
        constraints:
            List of :class:`SafetyConstraint` specifications to test.

        Returns
        -------
        list[SafetyViolation]
            All violations found.  Empty list means the action is safe.
        """
        violations: list[SafetyViolation] = []

        for constraint in constraints:
            violation = self._evaluate_constraint(action, constraint)
            if violation is not None:
                violations.append(violation)
                logger.debug(
                    "Safety violation: %s — %s", constraint.name, violation.message
                )

        return violations

    def _evaluate_constraint(
        self,
        action: list[float],
        constraint: SafetyConstraint,
    ) -> SafetyViolation | None:
        """Return a :class:`SafetyViolation` if the constraint is violated, else ``None``."""
        if constraint.constraint_type == ConstraintType.RANGE:
            return self._check_range(action, constraint)
        if constraint.constraint_type == ConstraintType.MAX_RATE:
            return self._check_max_rate(action, constraint)
        if constraint.constraint_type == ConstraintType.FORBIDDEN_ZONE:
            return self._check_forbidden_zone(action, constraint)
        # CUSTOM: callers are responsible for injecting their own logic;
        # the built-in checker cannot evaluate custom constraints.
        return None

    def _get_dimension_value(
        self, action: list[float], dimension: int
    ) -> float:
        """Return the action value for ``dimension``, or L2 norm when ``dimension == -1``."""
        if dimension == -1:
            return sum(v * v for v in action) ** 0.5
        if dimension < 0 or dimension >= len(action):
            raise IndexError(
                f"Constraint dimension {dimension} is out of range for "
                f"action of length {len(action)}."
            )
        return action[dimension]

    def _check_range(
        self, action: list[float], constraint: SafetyConstraint
    ) -> SafetyViolation | None:
        value = self._get_dimension_value(action, constraint.dimension)
        if value < constraint.min_value:
            return SafetyViolation(
                constraint_name=constraint.name,
                severity=constraint.severity,
                actual_value=value,
                message=(
                    f"Dimension {constraint.dimension} value {value:.6f} is below "
                    f"minimum {constraint.min_value:.6f}."
                ),
                dimension=constraint.dimension,
            )
        if value > constraint.max_value:
            return SafetyViolation(
                constraint_name=constraint.name,
                severity=constraint.severity,
                actual_value=value,
                message=(
                    f"Dimension {constraint.dimension} value {value:.6f} exceeds "
                    f"maximum {constraint.max_value:.6f}."
                ),
                dimension=constraint.dimension,
            )
        return None

    def _check_max_rate(
        self, action: list[float], constraint: SafetyConstraint
    ) -> SafetyViolation | None:
        if self._previous_action is None:
            return None
        current = self._get_dimension_value(action, constraint.dimension)
        previous = self._get_dimension_value(self._previous_action, constraint.dimension)
        delta = abs(current - previous)
        if delta > constraint.max_delta:
            return SafetyViolation(
                constraint_name=constraint.name,
                severity=constraint.severity,
                actual_value=delta,
                message=(
                    f"Dimension {constraint.dimension} rate of change {delta:.6f} exceeds "
                    f"maximum {constraint.max_delta:.6f}."
                ),
                dimension=constraint.dimension,
            )
        return None

    def _check_forbidden_zone(
        self, action: list[float], constraint: SafetyConstraint
    ) -> SafetyViolation | None:
        value = self._get_dimension_value(action, constraint.dimension)
        if constraint.min_value <= value <= constraint.max_value:
            return SafetyViolation(
                constraint_name=constraint.name,
                severity=constraint.severity,
                actual_value=value,
                message=(
                    f"Dimension {constraint.dimension} value {value:.6f} falls in "
                    f"forbidden zone [{constraint.min_value:.6f}, {constraint.max_value:.6f}]."
                ),
                dimension=constraint.dimension,
            )
        return None

    def __repr__(self) -> str:
        has_prev = self._previous_action is not None
        return f"SafetyChecker(has_previous_action={has_prev})"
