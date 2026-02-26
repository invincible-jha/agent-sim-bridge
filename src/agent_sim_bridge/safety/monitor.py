"""SafetyMonitor — real-time safety monitoring during episode execution.

The monitor wraps a :class:`~agent_sim_bridge.safety.constraints.SafetyChecker`
and maintains state across steps so violations can be accumulated and
analysed after the fact.  It also provides an emergency stop mechanism that
callers can activate when a CRITICAL violation is detected.

Thread safety
-------------
The monitor is *not* thread-safe by default.  If you need concurrent reads
of the violation log, wrap access with your own lock.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from agent_sim_bridge.safety.constraints import (
    SafetyChecker,
    SafetyConstraint,
    SafetyViolation,
    ViolationSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class MonitoredStep:
    """One monitored step record.

    Attributes
    ----------
    step_index:
        Zero-based step counter.
    action:
        The action that was checked.
    violations:
        All violations detected at this step.
    timestamp:
        Wall-clock time when the step was monitored.
    emergency_stop_triggered:
        True if this step triggered an emergency stop.
    """

    step_index: int
    action: list[float]
    violations: list[SafetyViolation]
    timestamp: float = field(default_factory=time.time)
    emergency_stop_triggered: bool = False


class SafetyMonitor:
    """Real-time safety monitor that accumulates violations across steps.

    Parameters
    ----------
    constraints:
        The safety constraints to enforce at every step.
    auto_stop_on_critical:
        When True (default) the monitor triggers an emergency stop
        automatically the first time a CRITICAL violation is seen.

    Example
    -------
    ::

        monitor = SafetyMonitor(constraints, auto_stop_on_critical=True)
        monitor.start_monitoring()

        while not done:
            action = agent.act(obs)
            violations = monitor.check_step(action)
            if monitor.emergency_stopped:
                break
            obs, reward, done, *_ = env.step(action)

        print(monitor.get_violations())
    """

    def __init__(
        self,
        constraints: list[SafetyConstraint],
        auto_stop_on_critical: bool = True,
    ) -> None:
        self._constraints = list(constraints)
        self._auto_stop_on_critical = auto_stop_on_critical
        self._checker = SafetyChecker()
        self._step_records: list[MonitoredStep] = []
        self._step_index: int = 0
        self._monitoring: bool = False
        self._emergency_stopped: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_monitoring(self) -> None:
        """Reset state and begin monitoring a new episode.

        Must be called before the first :meth:`check_step`.
        """
        self._step_records.clear()
        self._step_index = 0
        self._monitoring = True
        self._emergency_stopped = False
        self._checker = SafetyChecker()
        logger.info("SafetyMonitor started (%d constraints).", len(self._constraints))

    def stop_monitoring(self) -> None:
        """Stop monitoring.  Violation history is preserved."""
        self._monitoring = False
        logger.info(
            "SafetyMonitor stopped after %d steps, %d violation records.",
            self._step_index,
            self._count_total_violations(),
        )

    # ------------------------------------------------------------------
    # Per-step check
    # ------------------------------------------------------------------

    def check_step(self, action: list[float]) -> list[SafetyViolation]:
        """Check one action step and record any violations.

        Parameters
        ----------
        action:
            The action about to be sent to the environment.

        Returns
        -------
        list[SafetyViolation]
            Violations found at this step.  Empty when safe.

        Raises
        ------
        RuntimeError
            If :meth:`start_monitoring` has not been called, or if
            monitoring was stopped.
        """
        if not self._monitoring:
            raise RuntimeError(
                "SafetyMonitor is not active. Call start_monitoring() first."
            )
        if self._emergency_stopped:
            logger.warning(
                "check_step called after emergency stop at step %d.", self._step_index
            )

        violations = self._checker.check(action, self._constraints)
        emergency = False

        if violations and self._auto_stop_on_critical:
            for violation in violations:
                if violation.severity == ViolationSeverity.CRITICAL:
                    emergency = True
                    break

        record = MonitoredStep(
            step_index=self._step_index,
            action=list(action),
            violations=violations,
            emergency_stop_triggered=emergency,
        )
        self._step_records.append(record)
        self._checker.update_previous_action(action)
        self._step_index += 1

        if emergency:
            self.emergency_stop()

        return violations

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def emergency_stop(self) -> None:
        """Trigger an emergency stop.

        Sets :attr:`emergency_stopped` to True.  Callers should poll this
        flag after each step and halt actuation immediately.

        This method is idempotent — calling it a second time has no effect.
        """
        if not self._emergency_stopped:
            self._emergency_stopped = True
            logger.critical(
                "EMERGENCY STOP triggered at step %d.", self._step_index
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_violations(self) -> list[SafetyViolation]:
        """Return a flat list of all violations across all steps."""
        result: list[SafetyViolation] = []
        for record in self._step_records:
            result.extend(record.violations)
        return result

    def get_step_records(self) -> list[MonitoredStep]:
        """Return the full per-step monitoring history."""
        return list(self._step_records)

    @property
    def emergency_stopped(self) -> bool:
        """True if an emergency stop has been triggered."""
        return self._emergency_stopped

    @property
    def is_monitoring(self) -> bool:
        """True if the monitor is currently active."""
        return self._monitoring

    @property
    def step_count(self) -> int:
        """Number of steps checked since :meth:`start_monitoring`."""
        return self._step_index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_total_violations(self) -> int:
        return sum(len(r.violations) for r in self._step_records)

    def summary(self) -> dict[str, object]:
        """Return a plain-dict summary of monitoring results."""
        severity_counts: dict[str, int] = {s.value: 0 for s in ViolationSeverity}
        for violation in self.get_violations():
            severity_counts[violation.severity.value] += 1
        return {
            "total_steps": self._step_index,
            "total_violations": self._count_total_violations(),
            "emergency_stopped": self._emergency_stopped,
            "severity_counts": severity_counts,
        }

    def __repr__(self) -> str:
        return (
            f"SafetyMonitor("
            f"constraints={len(self._constraints)}, "
            f"monitoring={self._monitoring}, "
            f"emergency_stopped={self._emergency_stopped})"
        )
