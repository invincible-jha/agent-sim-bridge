"""Tests for agent_sim_bridge.cascade.simulator."""
from __future__ import annotations

import pytest

from agent_sim_bridge.cascade.simulator import (
    AgentNode,
    CascadeResult,
    CascadeSimulator,
    FailureMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_sim() -> CascadeSimulator:
    """Orchestrator → [tool_a, tool_b] → [downstream]."""
    sim = CascadeSimulator()
    sim.add_node(AgentNode("orchestrator", recovery_time_seconds=30, is_critical=True))
    sim.add_node(AgentNode("tool_a", recovery_time_seconds=10))
    sim.add_node(AgentNode("tool_b", recovery_time_seconds=5))
    sim.add_node(AgentNode("downstream", recovery_time_seconds=15))
    sim.add_dependency("orchestrator", "tool_a")
    sim.add_dependency("orchestrator", "tool_b")
    sim.add_dependency("tool_a", "downstream")
    return sim


@pytest.fixture()
def linear_sim() -> CascadeSimulator:
    """A → B → C → D."""
    sim = CascadeSimulator()
    for node_id in ["A", "B", "C", "D"]:
        sim.add_node(AgentNode(node_id, recovery_time_seconds=10))
    sim.add_dependency("A", "B")
    sim.add_dependency("B", "C")
    sim.add_dependency("C", "D")
    return sim


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_add_node(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("n1"))
        assert sim.node_count() == 1

    def test_get_node(self) -> None:
        sim = CascadeSimulator()
        node = AgentNode("n1", recovery_time_seconds=42.0)
        sim.add_node(node)
        assert sim.get_node("n1").recovery_time_seconds == 42.0

    def test_missing_node_raises_on_get(self) -> None:
        sim = CascadeSimulator()
        with pytest.raises(KeyError):
            sim.get_node("nonexistent")

    def test_add_dependency_unknown_dependency_raises(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("n1"))
        with pytest.raises(KeyError):
            sim.add_dependency("unknown", "n1")

    def test_add_dependency_unknown_dependent_raises(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("n1"))
        with pytest.raises(KeyError):
            sim.add_dependency("n1", "unknown")

    def test_node_count(self, simple_sim: CascadeSimulator) -> None:
        assert simple_sim.node_count() == 4


# ---------------------------------------------------------------------------
# Single node failure
# ---------------------------------------------------------------------------


class TestSingleNodeFailure:
    def test_isolated_node_blast_radius_100(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("only"))
        result = sim.simulate_failure("only")
        assert result.blast_radius_pct == 100.0

    def test_isolated_node_affected_is_self(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("only"))
        result = sim.simulate_failure("only")
        assert result.affected_node_ids == ["only"]

    def test_origin_always_affected(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("orchestrator")
        assert "orchestrator" in result.affected_node_ids


# ---------------------------------------------------------------------------
# Cascade propagation
# ---------------------------------------------------------------------------


class TestCascadePropagation:
    def test_orchestrator_failure_reaches_all(
        self, simple_sim: CascadeSimulator
    ) -> None:
        result = simple_sim.simulate_failure("orchestrator")
        # All 4 nodes should be affected
        assert len(result.affected_node_ids) == 4

    def test_leaf_node_no_cascade(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("downstream")
        # Only downstream itself is affected
        assert len(result.affected_node_ids) == 1

    def test_linear_chain_full_cascade(self, linear_sim: CascadeSimulator) -> None:
        result = linear_sim.simulate_failure("A")
        assert set(result.affected_node_ids) == {"A", "B", "C", "D"}

    def test_linear_chain_middle_failure(self, linear_sim: CascadeSimulator) -> None:
        result = linear_sim.simulate_failure("B")
        # B, C, D affected; A is upstream so unaffected
        assert "A" not in result.affected_node_ids
        assert "B" in result.affected_node_ids
        assert "D" in result.affected_node_ids

    def test_blast_radius_percentage(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("orchestrator")
        assert result.blast_radius_pct == 100.0

    def test_blast_radius_leaf(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("tool_b")
        assert result.blast_radius_pct == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Recovery time
# ---------------------------------------------------------------------------


class TestRecoveryTime:
    def test_recovery_is_max_of_affected(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("orchestrator")
        # orchestrator=30, tool_a=10, tool_b=5, downstream=15 → max=30
        assert result.estimated_recovery_seconds == 30.0

    def test_recovery_single_node(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("n", recovery_time_seconds=42.0))
        result = sim.simulate_failure("n")
        assert result.estimated_recovery_seconds == 42.0


# ---------------------------------------------------------------------------
# Critical nodes
# ---------------------------------------------------------------------------


class TestCriticalNodes:
    def test_critical_nodes_detected(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("orchestrator")
        assert "orchestrator" in result.critical_nodes_affected

    def test_no_critical_nodes_when_none_defined(
        self, linear_sim: CascadeSimulator
    ) -> None:
        result = linear_sim.simulate_failure("A")
        assert result.critical_nodes_affected == []


# ---------------------------------------------------------------------------
# Failure wave
# ---------------------------------------------------------------------------


class TestFailureWave:
    def test_failure_wave_has_entries(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("orchestrator")
        assert len(result.failure_wave) > 0

    def test_origin_is_wave_zero(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("orchestrator")
        wave_zero = [nid for wave_num, nid in result.failure_wave if wave_num == 0]
        assert "orchestrator" in wave_zero

    def test_linear_chain_wave_order(self, linear_sim: CascadeSimulator) -> None:
        result = linear_sim.simulate_failure("A")
        wave_map = {nid: wave_num for wave_num, nid in result.failure_wave}
        assert wave_map["A"] == 0
        assert wave_map["B"] == 1
        assert wave_map["C"] == 2
        assert wave_map["D"] == 3


# ---------------------------------------------------------------------------
# Propagation probability
# ---------------------------------------------------------------------------


class TestPropagationProbability:
    def test_zero_probability_does_not_propagate(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("A", propagation_probability=0.0))
        sim.add_node(AgentNode("B"))
        sim.add_dependency("A", "B")
        result = sim.simulate_failure("A")
        assert "B" not in result.affected_node_ids

    def test_full_probability_propagates(self) -> None:
        sim = CascadeSimulator()
        sim.add_node(AgentNode("A", propagation_probability=1.0))
        sim.add_node(AgentNode("B"))
        sim.add_dependency("A", "B")
        result = sim.simulate_failure("A")
        assert "B" in result.affected_node_ids


# ---------------------------------------------------------------------------
# Failure mode override
# ---------------------------------------------------------------------------


class TestFailureModeOverride:
    def test_failure_mode_override(self, simple_sim: CascadeSimulator) -> None:
        result = simple_sim.simulate_failure("orchestrator", failure_mode=FailureMode.TIMEOUT)
        # Should still cascade regardless of mode
        assert len(result.affected_node_ids) > 1


# ---------------------------------------------------------------------------
# Topological order
# ---------------------------------------------------------------------------


class TestTopologicalOrder:
    def test_topological_order_linear(self, linear_sim: CascadeSimulator) -> None:
        order = linear_sim.topological_order()
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")
        assert order.index("C") < order.index("D")

    def test_topological_order_contains_all_nodes(
        self, simple_sim: CascadeSimulator
    ) -> None:
        order = simple_sim.topological_order()
        assert set(order) == {"orchestrator", "tool_a", "tool_b", "downstream"}

    def test_reset_clears_failure_states(self, simple_sim: CascadeSimulator) -> None:
        simple_sim.simulate_failure("orchestrator")
        simple_sim.reset()
        for node_id in ["orchestrator", "tool_a", "tool_b", "downstream"]:
            assert simple_sim.get_node(node_id).failed is False

    def test_simulate_unknown_node_raises(self, simple_sim: CascadeSimulator) -> None:
        with pytest.raises(KeyError):
            simple_sim.simulate_failure("nonexistent")
