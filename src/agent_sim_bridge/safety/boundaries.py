"""BoundaryDefinition and BoundaryChecker — geometric boundary validation.

Boundaries define spatial regions (axis-aligned bounding boxes in
N-dimensional space) within which observations or positions must remain.
The checker validates that a given point is inside all registered boundaries.

Typical uses
------------
* Keep a robot's end-effector inside a safe workspace.
* Ensure a vehicle stays on the road surface.
* Restrict a simulation agent to a valid observation subspace.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BoundaryDefinition:
    """Axis-aligned bounding-box boundary in N-dimensional space.

    Attributes
    ----------
    name:
        Human-readable identifier for this boundary.
    lower_bounds:
        Per-dimension inclusive lower limits.  Must have the same length
        as ``upper_bounds``.
    upper_bounds:
        Per-dimension inclusive upper limits.
    description:
        Optional description of what this boundary represents.
    metadata:
        Arbitrary key/value annotations.

    Raises
    ------
    ValueError
        If ``lower_bounds`` and ``upper_bounds`` have different lengths,
        or if any lower bound exceeds its corresponding upper bound.
    """

    name: str
    lower_bounds: list[float]
    upper_bounds: list[float]
    description: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError(
                f"Boundary {self.name!r}: lower_bounds length "
                f"({len(self.lower_bounds)}) must equal upper_bounds length "
                f"({len(self.upper_bounds)})."
            )
        for i, (low, high) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            if low > high:
                raise ValueError(
                    f"Boundary {self.name!r}: lower_bounds[{i}]={low} "
                    f"exceeds upper_bounds[{i}]={high}."
                )

    @property
    def n_dims(self) -> int:
        """Number of dimensions in this boundary."""
        return len(self.lower_bounds)

    def contains(self, point: list[float]) -> bool:
        """Return True if ``point`` lies within the boundary (inclusive).

        Parameters
        ----------
        point:
            N-dimensional point to test.  Length must equal :attr:`n_dims`.

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            If ``point`` has the wrong dimensionality.
        """
        if len(point) != self.n_dims:
            raise ValueError(
                f"Boundary {self.name!r} is {self.n_dims}-dimensional, "
                f"but point has {len(point)} dimensions."
            )
        return all(
            low <= value <= high
            for value, low, high in zip(point, self.lower_bounds, self.upper_bounds)
        )

    def clamp(self, point: list[float]) -> list[float]:
        """Return ``point`` clamped to the boundary limits.

        Parameters
        ----------
        point:
            N-dimensional point to clamp.

        Returns
        -------
        list[float]
            Point with each coordinate clamped to ``[lower, upper]``.

        Raises
        ------
        ValueError
            If ``point`` has the wrong dimensionality.
        """
        if len(point) != self.n_dims:
            raise ValueError(
                f"Boundary {self.name!r} is {self.n_dims}-dimensional, "
                f"but point has {len(point)} dimensions."
            )
        return [
            max(low, min(value, high))
            for value, low, high in zip(point, self.lower_bounds, self.upper_bounds)
        ]


@dataclass
class BoundaryViolation:
    """Records a boundary violation for one point.

    Attributes
    ----------
    boundary_name:
        Name of the violated boundary.
    point:
        The point that was outside the boundary.
    violated_dimensions:
        Indices of dimensions where the point fell outside the boundary.
    message:
        Human-readable description.
    """

    boundary_name: str
    point: list[float]
    violated_dimensions: list[int]
    message: str


class BoundaryChecker:
    """Validate points against a collection of :class:`BoundaryDefinition` objects.

    Parameters
    ----------
    boundaries:
        Initial list of boundaries to enforce.  More can be added later via
        :meth:`add_boundary`.

    Example
    -------
    ::

        workspace = BoundaryDefinition(
            "workspace",
            lower_bounds=[-1.0, -1.0, 0.0],
            upper_bounds=[1.0, 1.0, 2.0],
        )
        checker = BoundaryChecker([workspace])
        violations = checker.check_point([0.5, 0.5, 0.5])  # empty — OK
        violations = checker.check_point([2.0, 0.0, 0.0])  # one violation
    """

    def __init__(self, boundaries: list[BoundaryDefinition] | None = None) -> None:
        self._boundaries: dict[str, BoundaryDefinition] = {}
        for boundary in boundaries or []:
            self.add_boundary(boundary)

    def add_boundary(self, boundary: BoundaryDefinition) -> None:
        """Register a boundary.

        Parameters
        ----------
        boundary:
            The :class:`BoundaryDefinition` to add.

        Raises
        ------
        ValueError
            If a boundary with the same name is already registered.
        """
        if boundary.name in self._boundaries:
            raise ValueError(
                f"Boundary {boundary.name!r} is already registered."
            )
        self._boundaries[boundary.name] = boundary
        logger.debug("Registered boundary %r (%d-D).", boundary.name, boundary.n_dims)

    def remove_boundary(self, name: str) -> None:
        """Remove a registered boundary by name.

        Raises
        ------
        KeyError
            If no boundary with that name is registered.
        """
        if name not in self._boundaries:
            raise KeyError(f"No boundary named {name!r} is registered.")
        del self._boundaries[name]

    def check_point(self, point: list[float]) -> list[BoundaryViolation]:
        """Check ``point`` against all registered boundaries.

        Only boundaries whose dimensionality matches ``len(point)`` are
        checked; mismatched boundaries are skipped with a warning.

        Parameters
        ----------
        point:
            The point to validate.

        Returns
        -------
        list[BoundaryViolation]
            All violations found.  Empty list means the point is within all
            applicable boundaries.
        """
        violations: list[BoundaryViolation] = []
        for boundary in self._boundaries.values():
            if boundary.n_dims != len(point):
                logger.warning(
                    "Boundary %r is %d-D but point is %d-D; skipping.",
                    boundary.name,
                    boundary.n_dims,
                    len(point),
                )
                continue
            if not boundary.contains(point):
                violated_dims = [
                    i
                    for i, (value, low, high) in enumerate(
                        zip(point, boundary.lower_bounds, boundary.upper_bounds)
                    )
                    if not (low <= value <= high)
                ]
                violation = BoundaryViolation(
                    boundary_name=boundary.name,
                    point=list(point),
                    violated_dimensions=violated_dims,
                    message=(
                        f"Point violates boundary {boundary.name!r} "
                        f"in dimensions {violated_dims}."
                    ),
                )
                violations.append(violation)
                logger.debug("Boundary violation: %s", violation.message)
        return violations

    def is_within_all_boundaries(self, point: list[float]) -> bool:
        """Return True only if the point violates no registered boundary.

        Parameters
        ----------
        point:
            The point to validate.
        """
        return len(self.check_point(point)) == 0

    def list_boundaries(self) -> list[str]:
        """Return sorted names of all registered boundaries."""
        return sorted(self._boundaries)

    def __len__(self) -> int:
        return len(self._boundaries)

    def __repr__(self) -> str:
        return f"BoundaryChecker(boundaries={self.list_boundaries()})"
