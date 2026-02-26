"""Unit tests for agent_sim_bridge.safety.boundaries.

Tests cover BoundaryDefinition construction, validation, contains(), clamp(),
and BoundaryChecker add/remove/check_point behaviour including dimension mismatch
handling and multi-boundary violations.
"""
from __future__ import annotations

import pytest

from agent_sim_bridge.safety.boundaries import (
    BoundaryChecker,
    BoundaryDefinition,
    BoundaryViolation,
)


# ---------------------------------------------------------------------------
# BoundaryDefinition — construction
# ---------------------------------------------------------------------------


class TestBoundaryDefinitionConstruction:
    def test_basic_construction_stores_attributes(self) -> None:
        boundary = BoundaryDefinition(
            name="workspace",
            lower_bounds=[-1.0, -1.0, 0.0],
            upper_bounds=[1.0, 1.0, 2.0],
        )

        assert boundary.name == "workspace"
        assert boundary.lower_bounds == [-1.0, -1.0, 0.0]
        assert boundary.upper_bounds == [1.0, 1.0, 2.0]

    def test_n_dims_matches_bounds_length(self) -> None:
        boundary = BoundaryDefinition(
            name="joint",
            lower_bounds=[-1.0, -2.0],
            upper_bounds=[1.0, 2.0],
        )

        assert boundary.n_dims == 2

    def test_one_dimensional_boundary(self) -> None:
        boundary = BoundaryDefinition(
            name="temperature",
            lower_bounds=[0.0],
            upper_bounds=[100.0],
        )

        assert boundary.n_dims == 1

    def test_description_and_metadata_defaults(self) -> None:
        boundary = BoundaryDefinition(
            name="test",
            lower_bounds=[0.0],
            upper_bounds=[1.0],
        )

        assert boundary.description == ""
        assert boundary.metadata == {}

    def test_description_and_metadata_set(self) -> None:
        boundary = BoundaryDefinition(
            name="test",
            lower_bounds=[0.0],
            upper_bounds=[1.0],
            description="A test boundary",
            metadata={"source": "robot_arm"},
        )

        assert boundary.description == "A test boundary"
        assert boundary.metadata["source"] == "robot_arm"

    def test_equal_lower_and_upper_is_valid(self) -> None:
        boundary = BoundaryDefinition(
            name="pinned",
            lower_bounds=[1.0, 1.0],
            upper_bounds=[1.0, 1.0],
        )

        assert boundary.n_dims == 2


# ---------------------------------------------------------------------------
# BoundaryDefinition — validation errors
# ---------------------------------------------------------------------------


class TestBoundaryDefinitionValidation:
    def test_mismatched_lengths_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="lower_bounds length"):
            BoundaryDefinition(
                name="bad",
                lower_bounds=[-1.0, -1.0],
                upper_bounds=[1.0],
            )

    def test_lower_exceeds_upper_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="lower_bounds\\[0\\]"):
            BoundaryDefinition(
                name="inverted",
                lower_bounds=[5.0],
                upper_bounds=[1.0],
            )

    def test_lower_exceeds_upper_in_second_dimension(self) -> None:
        with pytest.raises(ValueError, match="lower_bounds\\[1\\]"):
            BoundaryDefinition(
                name="partial_inversion",
                lower_bounds=[0.0, 10.0],
                upper_bounds=[1.0, 5.0],
            )

    def test_empty_bounds_are_valid(self) -> None:
        """Zero-dimensional boundaries are permitted (edge case)."""
        boundary = BoundaryDefinition(
            name="empty",
            lower_bounds=[],
            upper_bounds=[],
        )

        assert boundary.n_dims == 0


# ---------------------------------------------------------------------------
# BoundaryDefinition — contains()
# ---------------------------------------------------------------------------


class TestBoundaryDefinitionContains:
    def setup_method(self) -> None:
        self.boundary = BoundaryDefinition(
            name="workspace",
            lower_bounds=[-1.0, -1.0, 0.0],
            upper_bounds=[1.0, 1.0, 2.0],
        )

    def test_interior_point_returns_true(self) -> None:
        assert self.boundary.contains([0.0, 0.0, 1.0]) is True

    def test_point_on_lower_boundary_returns_true(self) -> None:
        assert self.boundary.contains([-1.0, -1.0, 0.0]) is True

    def test_point_on_upper_boundary_returns_true(self) -> None:
        assert self.boundary.contains([1.0, 1.0, 2.0]) is True

    def test_point_outside_first_dimension_returns_false(self) -> None:
        assert self.boundary.contains([2.0, 0.0, 1.0]) is False

    def test_point_below_lower_bound_returns_false(self) -> None:
        assert self.boundary.contains([-2.0, 0.0, 1.0]) is False

    def test_wrong_dimensionality_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="3-dimensional"):
            self.boundary.contains([0.0, 0.0])

    def test_point_outside_multiple_dimensions_returns_false(self) -> None:
        assert self.boundary.contains([5.0, 5.0, 5.0]) is False


# ---------------------------------------------------------------------------
# BoundaryDefinition — clamp()
# ---------------------------------------------------------------------------


class TestBoundaryDefinitionClamp:
    def setup_method(self) -> None:
        self.boundary = BoundaryDefinition(
            name="workspace",
            lower_bounds=[-1.0, -1.0],
            upper_bounds=[1.0, 1.0],
        )

    def test_interior_point_unchanged(self) -> None:
        result = self.boundary.clamp([0.5, -0.5])

        assert result == [0.5, -0.5]

    def test_value_above_upper_clamped_to_upper(self) -> None:
        result = self.boundary.clamp([5.0, 0.0])

        assert result[0] == pytest.approx(1.0)

    def test_value_below_lower_clamped_to_lower(self) -> None:
        result = self.boundary.clamp([-5.0, 0.0])

        assert result[0] == pytest.approx(-1.0)

    def test_both_dimensions_clamped(self) -> None:
        result = self.boundary.clamp([10.0, -10.0])

        assert result == [1.0, -1.0]

    def test_boundary_value_unchanged_after_clamp(self) -> None:
        result = self.boundary.clamp([1.0, -1.0])

        assert result == [1.0, -1.0]

    def test_wrong_dimensionality_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            self.boundary.clamp([0.0, 0.0, 0.0])

    def test_returns_new_list_not_original(self) -> None:
        point = [0.5, 0.5]
        result = self.boundary.clamp(point)

        assert result is not point


# ---------------------------------------------------------------------------
# BoundaryChecker — basic operations
# ---------------------------------------------------------------------------


class TestBoundaryCheckerBasicOperations:
    def test_empty_checker_has_zero_length(self) -> None:
        checker = BoundaryChecker()

        assert len(checker) == 0

    def test_add_boundary_increases_length(self) -> None:
        checker = BoundaryChecker()
        boundary = BoundaryDefinition("b", lower_bounds=[0.0], upper_bounds=[1.0])

        checker.add_boundary(boundary)

        assert len(checker) == 1

    def test_add_boundary_via_constructor(self) -> None:
        boundary = BoundaryDefinition("b", lower_bounds=[0.0], upper_bounds=[1.0])
        checker = BoundaryChecker([boundary])

        assert len(checker) == 1

    def test_list_boundaries_returns_sorted_names(self) -> None:
        checker = BoundaryChecker()
        checker.add_boundary(BoundaryDefinition("z_bound", lower_bounds=[0.0], upper_bounds=[1.0]))
        checker.add_boundary(BoundaryDefinition("a_bound", lower_bounds=[0.0], upper_bounds=[1.0]))

        names = checker.list_boundaries()

        assert names == ["a_bound", "z_bound"]

    def test_remove_boundary_decreases_length(self) -> None:
        boundary = BoundaryDefinition("b", lower_bounds=[0.0], upper_bounds=[1.0])
        checker = BoundaryChecker([boundary])

        checker.remove_boundary("b")

        assert len(checker) == 0

    def test_remove_nonexistent_boundary_raises_key_error(self) -> None:
        checker = BoundaryChecker()

        with pytest.raises(KeyError):
            checker.remove_boundary("missing")

    def test_add_duplicate_boundary_raises_value_error(self) -> None:
        boundary = BoundaryDefinition("b", lower_bounds=[0.0], upper_bounds=[1.0])
        checker = BoundaryChecker([boundary])

        with pytest.raises(ValueError, match="already registered"):
            checker.add_boundary(BoundaryDefinition("b", lower_bounds=[0.0], upper_bounds=[2.0]))

    def test_repr_contains_boundary_names(self) -> None:
        boundary = BoundaryDefinition("ws", lower_bounds=[0.0], upper_bounds=[1.0])
        checker = BoundaryChecker([boundary])

        assert "ws" in repr(checker)


# ---------------------------------------------------------------------------
# BoundaryChecker — check_point
# ---------------------------------------------------------------------------


class TestBoundaryCheckerCheckPoint:
    def setup_method(self) -> None:
        self.workspace = BoundaryDefinition(
            name="workspace",
            lower_bounds=[-1.0, -1.0, 0.0],
            upper_bounds=[1.0, 1.0, 2.0],
        )
        self.checker = BoundaryChecker([self.workspace])

    def test_point_inside_returns_no_violations(self) -> None:
        violations = self.checker.check_point([0.0, 0.0, 1.0])

        assert violations == []

    def test_point_outside_returns_one_violation(self) -> None:
        violations = self.checker.check_point([2.0, 0.0, 1.0])

        assert len(violations) == 1
        assert violations[0].boundary_name == "workspace"

    def test_violation_records_correct_point(self) -> None:
        violations = self.checker.check_point([2.0, 0.0, 1.0])

        assert violations[0].point == [2.0, 0.0, 1.0]

    def test_violation_identifies_correct_dimensions(self) -> None:
        violations = self.checker.check_point([2.0, 0.0, 1.0])

        assert 0 in violations[0].violated_dimensions

    def test_multiple_violated_dimensions_reported(self) -> None:
        violations = self.checker.check_point([5.0, -5.0, 1.0])

        violated_dims = violations[0].violated_dimensions
        assert 0 in violated_dims
        assert 1 in violated_dims

    def test_dimension_mismatch_skips_boundary(self) -> None:
        violations = self.checker.check_point([0.0, 0.0])

        assert violations == []

    def test_is_within_all_boundaries_returns_true_for_interior(self) -> None:
        assert self.checker.is_within_all_boundaries([0.0, 0.0, 1.0]) is True

    def test_is_within_all_boundaries_returns_false_for_exterior(self) -> None:
        assert self.checker.is_within_all_boundaries([5.0, 0.0, 1.0]) is False

    def test_multiple_boundaries_all_checked(self) -> None:
        second = BoundaryDefinition(
            name="speed_limit",
            lower_bounds=[-1.0, -1.0, 0.0],
            upper_bounds=[0.5, 0.5, 1.5],
        )
        self.checker.add_boundary(second)

        violations = self.checker.check_point([0.8, 0.0, 1.0])

        boundary_names = {v.boundary_name for v in violations}
        assert "speed_limit" in boundary_names

    def test_violation_message_is_non_empty(self) -> None:
        violations = self.checker.check_point([5.0, 0.0, 1.0])

        assert len(violations[0].message) > 0

    def test_boundary_violation_dataclass_attributes(self) -> None:
        violations = self.checker.check_point([5.0, 0.0, 1.0])
        violation = violations[0]

        assert isinstance(violation, BoundaryViolation)
        assert isinstance(violation.boundary_name, str)
        assert isinstance(violation.point, list)
        assert isinstance(violation.violated_dimensions, list)
        assert isinstance(violation.message, str)
