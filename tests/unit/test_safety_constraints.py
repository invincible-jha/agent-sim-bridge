"""Unit tests for agent_sim_bridge.safety.constraints.

Covers ConstraintType, ViolationSeverity enums, SafetyConstraint dataclass,
and SafetyChecker evaluation of RANGE, MAX_RATE, FORBIDDEN_ZONE, and
CUSTOM constraint types, including L2-norm (dimension=-1) checks.
"""
from __future__ import annotations

import pytest

from agent_sim_bridge.safety.constraints import (
    ConstraintType,
    SafetyChecker,
    SafetyConstraint,
    SafetyViolation,
    ViolationSeverity,
)


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


class TestConstraintTypeEnum:
    def test_range_value(self) -> None:
        assert ConstraintType.RANGE.value == "range"

    def test_max_rate_value(self) -> None:
        assert ConstraintType.MAX_RATE.value == "max_rate"

    def test_forbidden_zone_value(self) -> None:
        assert ConstraintType.FORBIDDEN_ZONE.value == "forbidden_zone"

    def test_custom_value(self) -> None:
        assert ConstraintType.CUSTOM.value == "custom"


class TestViolationSeverityEnum:
    def test_warning_value(self) -> None:
        assert ViolationSeverity.WARNING.value == "warning"

    def test_error_value(self) -> None:
        assert ViolationSeverity.ERROR.value == "error"

    def test_critical_value(self) -> None:
        assert ViolationSeverity.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# SafetyConstraint defaults
# ---------------------------------------------------------------------------


class TestSafetyConstraintDefaults:
    def test_default_constraint_type_is_range(self) -> None:
        constraint = SafetyConstraint(name="c")

        assert constraint.constraint_type == ConstraintType.RANGE

    def test_default_severity_is_error(self) -> None:
        constraint = SafetyConstraint(name="c")

        assert constraint.severity == ViolationSeverity.ERROR

    def test_default_dimension_is_zero(self) -> None:
        constraint = SafetyConstraint(name="c")

        assert constraint.dimension == 0

    def test_default_bounds_are_infinite(self) -> None:
        constraint = SafetyConstraint(name="c")

        assert constraint.min_value == float("-inf")
        assert constraint.max_value == float("inf")


# ---------------------------------------------------------------------------
# SafetyChecker — RANGE checks
# ---------------------------------------------------------------------------


class TestSafetyCheckerRange:
    def setup_method(self) -> None:
        self.checker = SafetyChecker()
        self.range_constraint = SafetyConstraint(
            name="joint_0_range",
            constraint_type=ConstraintType.RANGE,
            dimension=0,
            min_value=-1.0,
            max_value=1.0,
        )

    def test_action_within_range_returns_no_violations(self) -> None:
        violations = self.checker.check([0.5], [self.range_constraint])

        assert violations == []

    def test_action_at_lower_bound_returns_no_violations(self) -> None:
        violations = self.checker.check([-1.0], [self.range_constraint])

        assert violations == []

    def test_action_at_upper_bound_returns_no_violations(self) -> None:
        violations = self.checker.check([1.0], [self.range_constraint])

        assert violations == []

    def test_action_below_minimum_returns_violation(self) -> None:
        violations = self.checker.check([-2.0], [self.range_constraint])

        assert len(violations) == 1
        assert violations[0].constraint_name == "joint_0_range"

    def test_action_above_maximum_returns_violation(self) -> None:
        violations = self.checker.check([2.0], [self.range_constraint])

        assert len(violations) == 1

    def test_violation_carries_actual_value(self) -> None:
        violations = self.checker.check([-5.0], [self.range_constraint])

        assert violations[0].actual_value == pytest.approx(-5.0)

    def test_violation_carries_severity_from_constraint(self) -> None:
        critical_constraint = SafetyConstraint(
            name="critical",
            min_value=-1.0,
            max_value=1.0,
            severity=ViolationSeverity.CRITICAL,
        )
        violations = self.checker.check([5.0], [critical_constraint])

        assert violations[0].severity == ViolationSeverity.CRITICAL

    def test_multiple_constraints_all_evaluated(self) -> None:
        constraints = [
            SafetyConstraint(name="dim_0", dimension=0, min_value=-1.0, max_value=1.0),
            SafetyConstraint(name="dim_1", dimension=1, min_value=-1.0, max_value=1.0),
        ]

        violations = self.checker.check([2.0, 2.0], constraints)

        assert len(violations) == 2

    def test_only_violating_constraint_reported(self) -> None:
        constraints = [
            SafetyConstraint(name="dim_0", dimension=0, min_value=-1.0, max_value=1.0),
            SafetyConstraint(name="dim_1", dimension=1, min_value=-1.0, max_value=1.0),
        ]

        violations = self.checker.check([0.0, 2.0], constraints)

        assert len(violations) == 1
        assert violations[0].constraint_name == "dim_1"

    def test_l2_norm_range_check_dimension_minus_one(self) -> None:
        norm_constraint = SafetyConstraint(
            name="norm_limit",
            constraint_type=ConstraintType.RANGE,
            dimension=-1,
            min_value=0.0,
            max_value=1.0,
        )

        violations = self.checker.check([1.0, 1.0], [norm_constraint])

        assert len(violations) == 1

    def test_l2_norm_within_limit_no_violation(self) -> None:
        norm_constraint = SafetyConstraint(
            name="norm_limit",
            constraint_type=ConstraintType.RANGE,
            dimension=-1,
            min_value=0.0,
            max_value=2.0,
        )

        violations = self.checker.check([1.0, 0.0], [norm_constraint])

        assert violations == []

    def test_out_of_bounds_dimension_raises_index_error(self) -> None:
        bad_constraint = SafetyConstraint(name="bad", dimension=99, min_value=-1.0, max_value=1.0)

        with pytest.raises(IndexError):
            self.checker.check([0.5], [bad_constraint])

    def test_empty_constraints_list_returns_no_violations(self) -> None:
        violations = self.checker.check([0.5, -0.3], [])

        assert violations == []


# ---------------------------------------------------------------------------
# SafetyChecker — MAX_RATE checks
# ---------------------------------------------------------------------------


class TestSafetyCheckerMaxRate:
    def test_no_previous_action_returns_no_violation(self) -> None:
        checker = SafetyChecker(previous_action=None)
        constraint = SafetyConstraint(
            name="rate_limit",
            constraint_type=ConstraintType.MAX_RATE,
            dimension=0,
            max_delta=0.1,
        )

        violations = checker.check([1.0], [constraint])

        assert violations == []

    def test_small_delta_returns_no_violation(self) -> None:
        checker = SafetyChecker(previous_action=[0.5])
        constraint = SafetyConstraint(
            name="rate_limit",
            constraint_type=ConstraintType.MAX_RATE,
            dimension=0,
            max_delta=0.5,
        )

        violations = checker.check([0.6], [constraint])

        assert violations == []

    def test_large_delta_returns_violation(self) -> None:
        checker = SafetyChecker(previous_action=[0.0])
        constraint = SafetyConstraint(
            name="rate_limit",
            constraint_type=ConstraintType.MAX_RATE,
            dimension=0,
            max_delta=0.1,
        )

        violations = checker.check([1.0], [constraint])

        assert len(violations) == 1
        assert violations[0].constraint_name == "rate_limit"

    def test_update_previous_action_enables_rate_check(self) -> None:
        checker = SafetyChecker()
        constraint = SafetyConstraint(
            name="rate_limit",
            constraint_type=ConstraintType.MAX_RATE,
            dimension=0,
            max_delta=0.1,
        )

        checker.update_previous_action([0.0])
        violations = checker.check([1.0], [constraint])

        assert len(violations) == 1

    def test_delta_violation_carries_delta_as_actual_value(self) -> None:
        checker = SafetyChecker(previous_action=[0.0])
        constraint = SafetyConstraint(
            name="rate",
            constraint_type=ConstraintType.MAX_RATE,
            dimension=0,
            max_delta=0.1,
        )

        violations = checker.check([0.5], [constraint])

        assert violations[0].actual_value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SafetyChecker — FORBIDDEN_ZONE checks
# ---------------------------------------------------------------------------


class TestSafetyCheckerForbiddenZone:
    def setup_method(self) -> None:
        self.checker = SafetyChecker()
        self.forbidden_constraint = SafetyConstraint(
            name="dead_zone",
            constraint_type=ConstraintType.FORBIDDEN_ZONE,
            dimension=0,
            min_value=-0.1,
            max_value=0.1,
        )

    def test_value_inside_forbidden_zone_returns_violation(self) -> None:
        violations = self.checker.check([0.0], [self.forbidden_constraint])

        assert len(violations) == 1

    def test_value_at_zone_boundary_returns_violation(self) -> None:
        violations = self.checker.check([0.1], [self.forbidden_constraint])

        assert len(violations) == 1

    def test_value_outside_forbidden_zone_returns_no_violation(self) -> None:
        violations = self.checker.check([0.5], [self.forbidden_constraint])

        assert violations == []

    def test_value_below_zone_returns_no_violation(self) -> None:
        violations = self.checker.check([-0.5], [self.forbidden_constraint])

        assert violations == []


# ---------------------------------------------------------------------------
# SafetyChecker — CUSTOM constraint (pass-through)
# ---------------------------------------------------------------------------


class TestSafetyCheckerCustomConstraint:
    def test_custom_constraint_not_evaluated_by_builtin_checker(self) -> None:
        checker = SafetyChecker()
        custom = SafetyConstraint(
            name="custom_logic",
            constraint_type=ConstraintType.CUSTOM,
            dimension=0,
        )

        violations = checker.check([999.0], [custom])

        assert violations == []


# ---------------------------------------------------------------------------
# SafetyChecker — repr
# ---------------------------------------------------------------------------


class TestSafetyCheckerRepr:
    def test_repr_without_previous_action(self) -> None:
        checker = SafetyChecker()

        assert "False" in repr(checker)

    def test_repr_with_previous_action(self) -> None:
        checker = SafetyChecker(previous_action=[1.0])

        assert "True" in repr(checker)


# ---------------------------------------------------------------------------
# SafetyViolation dataclass
# ---------------------------------------------------------------------------


class TestSafetyViolation:
    def test_violation_attributes_accessible(self) -> None:
        violation = SafetyViolation(
            constraint_name="test",
            severity=ViolationSeverity.ERROR,
            actual_value=5.0,
            message="value exceeded limit",
            dimension=0,
        )

        assert violation.constraint_name == "test"
        assert violation.severity == ViolationSeverity.ERROR
        assert violation.actual_value == pytest.approx(5.0)
        assert violation.message == "value exceeded limit"
        assert violation.dimension == 0
