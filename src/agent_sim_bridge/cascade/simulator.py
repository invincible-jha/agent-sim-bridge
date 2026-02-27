"""Failure cascade simulator for agent dependency graphs.

Models agent components as nodes in a directed acyclic graph (DAG).
A failure injected at one node propagates to all downstream dependents.
The simulator reports the blast radius (percentage of nodes affected)
and an estimated recovery time based on each node's recovery profile.

Usage
-----
::

    sim = CascadeSimulator()
    sim.add_node(AgentNode("orchestrator", recovery_time_seconds=30))
    sim.add_node(AgentNode("tool_a", recovery_time_seconds=10))
    sim.add_node(AgentNode("tool_b", recovery_time_seconds=5))
    sim.add_dependency("orchestrator", "tool_a")
    sim.add_dependency("orchestrator", "tool_b")
    result = sim.simulate_failure("orchestrator")
    print(result.blast_radius_pct)
"""
from __future__ import annotations

import datetime
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FailureMode(str, Enum):
    """Type of failure injected at a node."""

    CRASH = "crash"           # Node stops responding
    DEGRADED = "degraded"     # Node responds slowly / with errors
    TIMEOUT = "timeout"       # Node times out; dependents stall
    DEPENDENCY_ERROR = "dependency_error"  # Propagated failure from upstream


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


@dataclass
class AgentNode:
    """A node in the agent dependency DAG.

    Attributes
    ----------
    node_id:
        Unique identifier for this agent component.
    recovery_time_seconds:
        Estimated time in seconds for this node to recover from a failure.
    is_critical:
        Whether failure of this node constitutes a system-critical event.
    propagation_probability:
        Probability [0.0, 1.0] that this node's failure propagates to each
        dependent.  1.0 = always propagates (hard dependency).
    failure_mode:
        The type of failure this node exhibits when it fails.
    metadata:
        Optional additional attributes.
    failed:
        Whether this node is currently in a failed state.
    """

    node_id: str
    recovery_time_seconds: float = 30.0
    is_critical: bool = False
    propagation_probability: float = 1.0
    failure_mode: FailureMode = FailureMode.CRASH
    metadata: dict[str, object] = field(default_factory=dict)
    failed: bool = False


# ---------------------------------------------------------------------------
# Cascade result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeResult:
    """Result of a cascade failure simulation.

    Attributes
    ----------
    origin_node_id:
        The node where the failure was injected.
    affected_node_ids:
        IDs of all nodes that were affected (including origin).
    total_node_count:
        Total number of nodes in the graph.
    blast_radius_pct:
        Percentage of nodes affected: len(affected) / total * 100.
    estimated_recovery_seconds:
        Maximum recovery time across affected nodes (the bottleneck).
    critical_nodes_affected:
        IDs of critical nodes that were impacted.
    failure_wave:
        Ordered list of (wave_number, node_id) showing how the failure
        propagated through the graph.
    simulated_at:
        UTC timestamp when the simulation was run.
    """

    origin_node_id: str
    affected_node_ids: list[str]
    total_node_count: int
    blast_radius_pct: float
    estimated_recovery_seconds: float
    critical_nodes_affected: list[str]
    failure_wave: list[tuple[int, str]]
    simulated_at: datetime.datetime


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class CascadeSimulator:
    """Simulate failure cascades in an agent dependency DAG.

    The graph is directed: an edge A → B means B *depends on* A.
    When A fails, the failure propagates to B and transitively to all of
    B's dependents.

    Example
    -------
    ::

        sim = CascadeSimulator()
        sim.add_node(AgentNode("A", recovery_time_seconds=20, is_critical=True))
        sim.add_node(AgentNode("B", recovery_time_seconds=10))
        sim.add_dependency("A", "B")  # B depends on A
        result = sim.simulate_failure("A")
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AgentNode] = {}
        # _dependents[X] = list of nodes that depend on X
        self._dependents: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: AgentNode) -> None:
        """Add an agent node to the dependency graph.

        Parameters
        ----------
        node:
            The :class:`AgentNode` to add.
        """
        self._nodes[node.node_id] = node
        if node.node_id not in self._dependents:
            self._dependents[node.node_id] = []

    def add_dependency(self, dependency_id: str, dependent_id: str) -> None:
        """Declare that *dependent_id* depends on *dependency_id*.

        A failure in *dependency_id* will propagate to *dependent_id*.

        Parameters
        ----------
        dependency_id:
            The upstream node (the one that can fail).
        dependent_id:
            The downstream node (the one affected when upstream fails).

        Raises
        ------
        KeyError
            If either node has not been added via :meth:`add_node`.
        """
        if dependency_id not in self._nodes:
            raise KeyError(f"Node '{dependency_id}' not in graph")
        if dependent_id not in self._nodes:
            raise KeyError(f"Node '{dependent_id}' not in graph")
        if dependent_id not in self._dependents[dependency_id]:
            self._dependents[dependency_id].append(dependent_id)

    def node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        return len(self._nodes)

    def get_node(self, node_id: str) -> AgentNode:
        """Return the node with *node_id*.

        Raises
        ------
        KeyError
            If the node is not in the graph.
        """
        return self._nodes[node_id]

    # ------------------------------------------------------------------
    # Failure simulation
    # ------------------------------------------------------------------

    def simulate_failure(
        self,
        origin_node_id: str,
        failure_mode: FailureMode | None = None,
    ) -> CascadeResult:
        """Inject a failure at *origin_node_id* and propagate it.

        Uses breadth-first traversal over the dependency graph.  Each
        node's ``propagation_probability`` is treated as a hard threshold
        for deterministic testing (probability=1.0 means always propagate;
        probability=0.0 means never propagates beyond that node).

        Parameters
        ----------
        origin_node_id:
            The node where the failure originates.
        failure_mode:
            Override the origin node's default failure mode.

        Returns
        -------
        CascadeResult
            Full cascade simulation result.

        Raises
        ------
        KeyError
            If *origin_node_id* is not in the graph.
        """
        if origin_node_id not in self._nodes:
            raise KeyError(f"Node '{origin_node_id}' not in graph")

        # Reset failure states
        for node in self._nodes.values():
            node.failed = False

        affected: list[str] = []
        failure_wave: list[tuple[int, str]] = []
        visited: set[str] = set()

        queue: deque[tuple[str, int]] = deque()
        queue.append((origin_node_id, 0))
        visited.add(origin_node_id)

        origin_node = self._nodes[origin_node_id]
        if failure_mode is not None:
            actual_mode = failure_mode
        else:
            actual_mode = origin_node.failure_mode

        while queue:
            current_id, wave = queue.popleft()
            current = self._nodes[current_id]
            current.failed = True
            affected.append(current_id)
            failure_wave.append((wave, current_id))

            # Propagate to dependents
            for dep_id in self._dependents.get(current_id, []):
                if dep_id in visited:
                    continue
                dep_node = self._nodes[dep_id]
                # Propagate based on probability (deterministic: >= 0.5 propagates)
                if current.propagation_probability >= 0.5:
                    visited.add(dep_id)
                    queue.append((dep_id, wave + 1))

        total_nodes = len(self._nodes)
        blast_radius = (len(affected) / total_nodes * 100.0) if total_nodes > 0 else 0.0

        # Estimated recovery = max recovery time among affected nodes
        recovery_times = [
            self._nodes[nid].recovery_time_seconds for nid in affected
        ]
        estimated_recovery = max(recovery_times) if recovery_times else 0.0

        critical_affected = [
            nid for nid in affected if self._nodes[nid].is_critical
        ]

        return CascadeResult(
            origin_node_id=origin_node_id,
            affected_node_ids=affected,
            total_node_count=total_nodes,
            blast_radius_pct=round(blast_radius, 2),
            estimated_recovery_seconds=estimated_recovery,
            critical_nodes_affected=critical_affected,
            failure_wave=failure_wave,
            simulated_at=datetime.datetime.now(datetime.timezone.utc),
        )

    def reset(self) -> None:
        """Reset all node failure states."""
        for node in self._nodes.values():
            node.failed = False

    def topological_order(self) -> list[str]:
        """Return nodes in topological order (roots first).

        Uses Kahn's algorithm.  Raises ValueError on cycle detection.

        Returns
        -------
        list[str]
            Node IDs in topological order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for node_id, deps in self._dependents.items():
            for dep_id in deps:
                in_degree[dep_id] = in_degree.get(dep_id, 0) + 1

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for dep_id in self._dependents.get(current, []):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if len(order) != len(self._nodes):
            raise ValueError("Dependency graph contains a cycle")
        return order


__all__ = [
    "AgentNode",
    "CascadeResult",
    "CascadeSimulator",
    "FailureMode",
]
