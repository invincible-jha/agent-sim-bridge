"""Simulated user profiles for agent staging environments.

Models a synthetic user that sends messages sequentially and evaluates
agent responses against expected outcomes.  The evaluation is a simple
keyword-match heuristic — sufficient for staging smoke tests without
exposing ML-based quality scoring.

Classes
-------
- UserProfile    Frozen, immutable configuration for a simulated user.
- SimulatedUser  Stateful user simulation that advances through messages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserProfile:
    """Immutable profile defining a simulated user's behaviour.

    Attributes
    ----------
    name:
        Human-readable name for this user profile (e.g. ``"alice"``).
    persona:
        Short label describing the user's behaviour style.
        Common values: ``"impatient"``, ``"technical"``, ``"confused"``,
        ``"friendly"``.  Used for logging and reporting only.
    messages:
        Ordered tuple of messages the user will send, one per turn.
    expected_outcomes:
        Tuple of expected keywords or phrases that a satisfactory
        response should contain.  Each entry is checked against the
        agent response; the fraction matched determines satisfaction.
    """

    name: str
    persona: str
    messages: tuple[str, ...]
    expected_outcomes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UserProfile.name must not be empty.")
        if not self.persona:
            raise ValueError("UserProfile.persona must not be empty.")


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------


class SimulatedUser:
    """Simulates user behaviour according to a :class:`UserProfile`.

    Maintains a cursor position into ``profile.messages`` so that
    successive calls to :meth:`next_message` advance through the
    conversation.  When all messages are exhausted, :meth:`next_message`
    returns ``None``.

    Parameters
    ----------
    profile:
        The immutable user profile driving this simulation.
    """

    def __init__(self, profile: UserProfile) -> None:
        self._profile = profile
        self._cursor: int = 0
        self._satisfaction_scores: list[float] = []

    @property
    def profile(self) -> UserProfile:
        """The immutable user profile."""
        return self._profile

    @property
    def name(self) -> str:
        """User's name (from profile)."""
        return self._profile.name

    @property
    def persona(self) -> str:
        """User's persona label (from profile)."""
        return self._profile.persona

    @property
    def messages_sent(self) -> int:
        """Number of messages the user has sent so far."""
        return self._cursor

    @property
    def is_done(self) -> bool:
        """Return True if all messages have been sent."""
        return self._cursor >= len(self._profile.messages)

    def next_message(self) -> str | None:
        """Return the next message in sequence, or ``None`` when exhausted.

        Advances the internal cursor on each call.

        Returns
        -------
        str | None
            The next user message, or ``None`` if all messages have
            been sent.
        """
        if self._cursor >= len(self._profile.messages):
            return None
        message = self._profile.messages[self._cursor]
        self._cursor += 1
        logger.debug(
            "SimulatedUser[%s]: sending message %d/%d: %r",
            self._profile.name,
            self._cursor,
            len(self._profile.messages),
            message,
        )
        return message

    def evaluate_response(self, response: str) -> float:
        """Score *response* against the user's expected outcomes.

        Computes a satisfaction score in [0.0, 1.0] by counting how many
        ``expected_outcomes`` keywords appear in *response* (case-insensitive).
        If no expected outcomes are defined, returns 1.0 (fully satisfied).

        Parameters
        ----------
        response:
            The agent's textual response to evaluate.

        Returns
        -------
        float
            Satisfaction score: 1.0 = all outcomes found, 0.0 = none found.
        """
        expected = self._profile.expected_outcomes
        if not expected:
            score = 1.0
        else:
            response_lower = response.lower()
            matched = sum(
                1
                for outcome in expected
                if outcome.lower() in response_lower
            )
            score = matched / len(expected)

        self._satisfaction_scores.append(score)
        logger.debug(
            "SimulatedUser[%s]: satisfaction=%.2f for response=%r",
            self._profile.name,
            score,
            response[:80],
        )
        return score

    def average_satisfaction(self) -> float:
        """Return the mean satisfaction score across all evaluated responses.

        Returns 0.0 if no responses have been evaluated.

        Returns
        -------
        float
            Average satisfaction score in [0.0, 1.0].
        """
        if not self._satisfaction_scores:
            return 0.0
        return sum(self._satisfaction_scores) / len(self._satisfaction_scores)

    def reset(self) -> None:
        """Reset the user's cursor and satisfaction history to initial state."""
        self._cursor = 0
        self._satisfaction_scores = []
        logger.debug("SimulatedUser[%s]: reset.", self._profile.name)

    def __repr__(self) -> str:
        return (
            f"SimulatedUser("
            f"name={self._profile.name!r}, "
            f"persona={self._profile.persona!r}, "
            f"cursor={self._cursor}/{len(self._profile.messages)}"
            f")"
        )


__all__ = [
    "UserProfile",
    "SimulatedUser",
]
