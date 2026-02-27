"""Full agent staging environment with simulated users, tools, and chaos.

The :class:`StagingEnvironment` wires together simulated users, tools,
and a chaos engine into a single harness.  An agent function is called
once per scenario and its responses are scored by each simulated user.
The scenario result captures pass/fail, satisfaction, timing, and chaos
events.

Classes
-------
- StagingReport       Immutable aggregate of all scenario results.
- StagingEnvironment  Main harness for running staging scenarios.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from agent_sim_bridge.staging.chaos import ChaosConfig, ChaosEngine
from agent_sim_bridge.staging.results import StagingTestResult
from agent_sim_bridge.staging.simulated_tools import SimulatedTool, ToolBehavior
from agent_sim_bridge.staging.simulated_users import SimulatedUser, UserProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StagingReport:
    """Immutable aggregate report for a set of staging scenario results.

    Attributes
    ----------
    total_scenarios:
        Total number of scenarios that were executed.
    passed:
        Number of scenarios that passed.
    failed:
        Number of scenarios that failed.
    average_satisfaction:
        Mean user satisfaction score across all scenarios (0.0–1.0).
    average_latency_ms:
        Mean scenario duration across all scenarios (milliseconds).
    chaos_events_total:
        Total chaos events across all scenarios.
    results:
        Tuple of all individual :class:`StagingTestResult` instances.
    """

    total_scenarios: int
    passed: int
    failed: int
    average_satisfaction: float
    average_latency_ms: float
    chaos_events_total: int
    results: tuple[StagingTestResult, ...]

    def __post_init__(self) -> None:
        if self.total_scenarios < 0:
            raise ValueError("total_scenarios must be >= 0.")
        if self.passed + self.failed > self.total_scenarios:
            raise ValueError(
                f"passed ({self.passed}) + failed ({self.failed}) "
                f"exceeds total_scenarios ({self.total_scenarios})."
            )

    @property
    def pass_rate(self) -> float:
        """Fraction of scenarios that passed (0.0–1.0).

        Returns 0.0 if no scenarios were run.
        """
        if self.total_scenarios == 0:
            return 0.0
        return self.passed / self.total_scenarios


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class StagingEnvironment:
    """Full agent staging environment with simulated users, tools, and chaos.

    Usage::

        env = StagingEnvironment(chaos_config=ChaosConfig(latency_injection_ms=50.0))
        env.add_user(UserProfile(name="alice", persona="technical", ...))
        env.add_tool(ToolBehavior(name="search", success_rate=0.9, latency_ms=20.0))

        async def my_agent(message: str, tools: dict) -> str:
            ...

        result = asyncio.run(env.run_scenario(my_agent, "smoke-test"))
        report = env.aggregate_results([result])

    Parameters
    ----------
    chaos_config:
        Optional chaos configuration.  If ``None``, chaos is disabled
        (all rates = 0, no latency).
    pass_threshold:
        Minimum average user satisfaction required for a scenario to
        pass.  Defaults to 0.5.
    """

    def __init__(
        self,
        chaos_config: ChaosConfig | None = None,
        pass_threshold: float = 0.5,
    ) -> None:
        self._chaos_config = chaos_config or ChaosConfig()
        self._chaos_engine = ChaosEngine(self._chaos_config)
        self._pass_threshold = pass_threshold
        self._users: list[SimulatedUser] = []
        self._tools: dict[str, SimulatedTool] = {}

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------

    def add_user(self, profile: UserProfile) -> None:
        """Register a simulated user in this staging environment.

        Parameters
        ----------
        profile:
            The user profile to add.
        """
        user = SimulatedUser(profile)
        self._users.append(user)
        logger.debug(
            "StagingEnvironment: added user %r (persona=%r).",
            profile.name,
            profile.persona,
        )

    def add_tool(self, behavior: ToolBehavior) -> None:
        """Register a simulated tool in this staging environment.

        Parameters
        ----------
        behavior:
            The tool behaviour to add.  The tool is keyed by
            ``behavior.name``.  Adding a tool with a duplicate name
            replaces the previous one.
        """
        self._tools[behavior.name] = SimulatedTool(
            behavior,
            random_seed=self._chaos_config.random_seed,
        )
        logger.debug(
            "StagingEnvironment: added tool %r (success_rate=%.2f).",
            behavior.name,
            behavior.success_rate,
        )

    @property
    def user_count(self) -> int:
        """Number of simulated users registered."""
        return len(self._users)

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._tools)

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    async def run_scenario(
        self,
        agent_fn: Callable[...],
        scenario_name: str,
    ) -> StagingTestResult:
        """Execute *agent_fn* against all registered users and record results.

        The agent function is called with each user's messages in sequence.
        After each agent response the user's satisfaction is recorded.
        Chaos events are applied per-call.

        The agent function signature must be::

            async def agent_fn(
                message: str,
                tools: dict[str, SimulatedTool],
                context: dict[str, object],
            ) -> str: ...

        or a synchronous equivalent (detected automatically).

        Parameters
        ----------
        agent_fn:
            The agent function to test.
        scenario_name:
            Human-readable label for this scenario run.

        Returns
        -------
        StagingTestResult
            Immutable result record for this scenario.
        """
        start = time.monotonic()
        chaos_events: list[str] = []
        errors: list[str] = []
        satisfaction_scores: list[float] = []

        # Reset users before the scenario
        for user in self._users:
            user.reset()

        self._chaos_engine.reset_counters()

        # Run interactions
        for user in self._users:
            while not user.is_done:
                message = user.next_message()
                if message is None:
                    break

                # Check for network partition
                if self._chaos_engine.should_partition():
                    chaos_events.append("network_partition")
                    errors.append(
                        f"Network partition for user={user.name!r} msg={message!r}"
                    )
                    # Treat as a zero-satisfaction response
                    satisfaction_scores.append(0.0)
                    continue

                # Apply latency
                injected_latency = self._chaos_engine.inject_latency()
                if injected_latency > 0:
                    chaos_events.append(f"latency_injected_{injected_latency:.0f}ms")
                    await asyncio.sleep(injected_latency / 1000.0)

                # Call the agent
                try:
                    context: dict[str, object] = {
                        "user": user.name,
                        "persona": user.persona,
                        "scenario": scenario_name,
                    }
                    if asyncio.iscoroutinefunction(agent_fn):
                        response = await agent_fn(
                            message=message,
                            tools=self._tools,
                            context=context,
                        )
                    else:
                        response = agent_fn(
                            message=message,
                            tools=self._tools,
                            context=context,
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"Agent error for user={user.name!r}: {exc}"
                    )
                    satisfaction_scores.append(0.0)
                    continue

                # Degrade output if configured
                if isinstance(response, str):
                    degraded = self._chaos_engine.degrade_output(response)
                    if degraded != response:
                        chaos_events.append("model_output_degraded")
                    response = degraded

                # Evaluate response
                score = user.evaluate_response(str(response))
                satisfaction_scores.append(score)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        avg_satisfaction = (
            sum(satisfaction_scores) / len(satisfaction_scores)
            if satisfaction_scores
            else 0.0
        )
        passed = avg_satisfaction >= self._pass_threshold and not any(
            "network_partition" in e for e in chaos_events
        )

        result = StagingTestResult(
            test_name=scenario_name,
            passed=passed,
            duration_ms=elapsed_ms,
            user_satisfaction=avg_satisfaction,
            chaos_events=tuple(chaos_events),
            errors=tuple(errors),
        )

        logger.info(
            "StagingEnvironment: scenario=%r passed=%s satisfaction=%.2f (%.2f ms).",
            scenario_name,
            passed,
            avg_satisfaction,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(
        self,
        results: list[StagingTestResult],
    ) -> StagingReport:
        """Aggregate a list of :class:`StagingTestResult` into a report.

        Parameters
        ----------
        results:
            List of scenario results to aggregate.

        Returns
        -------
        StagingReport
            Immutable aggregate report.
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        avg_satisfaction = (
            sum(r.user_satisfaction for r in results) / total
            if total > 0
            else 0.0
        )
        avg_latency = (
            sum(r.duration_ms for r in results) / total
            if total > 0
            else 0.0
        )
        chaos_total = sum(r.chaos_event_count for r in results)

        report = StagingReport(
            total_scenarios=total,
            passed=passed,
            failed=failed,
            average_satisfaction=avg_satisfaction,
            average_latency_ms=avg_latency,
            chaos_events_total=chaos_total,
            results=tuple(results),
        )
        logger.info(
            "StagingEnvironment: aggregate report — %d/%d passed (%.2f%%).",
            passed,
            total,
            report.pass_rate * 100,
        )
        return report

    def __repr__(self) -> str:
        return (
            f"StagingEnvironment("
            f"users={len(self._users)}, "
            f"tools={len(self._tools)}, "
            f"chaos={self._chaos_config!r}"
            f")"
        )


__all__ = [
    "StagingReport",
    "StagingEnvironment",
]
