"""Tests for simulated user profiles and behaviour.

Covers:
- UserProfile validation
- SimulatedUser.next_message() message sequence
- SimulatedUser.is_done / messages_sent
- SimulatedUser.evaluate_response() satisfaction scoring
- SimulatedUser.average_satisfaction()
- SimulatedUser.reset()
- Edge cases: no expected outcomes, empty messages
- repr()
"""
from __future__ import annotations

import pytest

from agent_sim_bridge.staging.simulated_users import SimulatedUser, UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(
    name: str = "alice",
    persona: str = "technical",
    messages: tuple[str, ...] = ("Hello", "How are you?"),
    expected_outcomes: tuple[str, ...] = ("fine", "good"),
) -> UserProfile:
    return UserProfile(
        name=name,
        persona=persona,
        messages=messages,
        expected_outcomes=expected_outcomes,
    )


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------


class TestUserProfile:
    def test_valid_profile(self) -> None:
        profile = _make_profile()
        assert profile.name == "alice"
        assert profile.persona == "technical"

    def test_frozen(self) -> None:
        profile = _make_profile()
        with pytest.raises((AttributeError, TypeError)):
            profile.name = "bob"  # type: ignore[misc]

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            UserProfile(
                name="",
                persona="technical",
                messages=("hi",),
                expected_outcomes=(),
            )

    def test_empty_persona_raises(self) -> None:
        with pytest.raises(ValueError, match="persona must not be empty"):
            UserProfile(
                name="alice",
                persona="",
                messages=("hi",),
                expected_outcomes=(),
            )

    def test_empty_messages_allowed(self) -> None:
        profile = UserProfile(
            name="silent",
            persona="quiet",
            messages=(),
            expected_outcomes=(),
        )
        assert profile.messages == ()

    def test_empty_expected_outcomes_allowed(self) -> None:
        profile = _make_profile(expected_outcomes=())
        assert profile.expected_outcomes == ()


# ---------------------------------------------------------------------------
# SimulatedUser.next_message()
# ---------------------------------------------------------------------------


class TestSimulatedUserNextMessage:
    def test_first_message_returned(self) -> None:
        user = SimulatedUser(_make_profile(messages=("Hello", "Bye")))
        assert user.next_message() == "Hello"

    def test_second_message_returned(self) -> None:
        user = SimulatedUser(_make_profile(messages=("Hello", "Bye")))
        user.next_message()
        assert user.next_message() == "Bye"

    def test_none_when_exhausted(self) -> None:
        user = SimulatedUser(_make_profile(messages=("Only message",)))
        user.next_message()
        assert user.next_message() is None

    def test_messages_sent_increments(self) -> None:
        user = SimulatedUser(_make_profile(messages=("A", "B", "C")))
        assert user.messages_sent == 0
        user.next_message()
        assert user.messages_sent == 1
        user.next_message()
        assert user.messages_sent == 2

    def test_is_done_false_initially(self) -> None:
        user = SimulatedUser(_make_profile(messages=("Hello",)))
        assert user.is_done is False

    def test_is_done_true_after_exhaustion(self) -> None:
        user = SimulatedUser(_make_profile(messages=("Hello",)))
        user.next_message()
        assert user.is_done is True

    def test_empty_messages_is_done_immediately(self) -> None:
        user = SimulatedUser(_make_profile(messages=()))
        assert user.is_done is True
        assert user.next_message() is None


# ---------------------------------------------------------------------------
# SimulatedUser.evaluate_response()
# ---------------------------------------------------------------------------


class TestSimulatedUserEvaluateResponse:
    def test_all_outcomes_present_returns_one(self) -> None:
        user = SimulatedUser(
            _make_profile(expected_outcomes=("hello", "world"))
        )
        score = user.evaluate_response("Hello World example")
        assert score == pytest.approx(1.0)

    def test_no_outcomes_present_returns_zero(self) -> None:
        user = SimulatedUser(
            _make_profile(expected_outcomes=("sunshine", "rainbow"))
        )
        score = user.evaluate_response("It is raining today")
        assert score == pytest.approx(0.0)

    def test_half_outcomes_present_returns_half(self) -> None:
        user = SimulatedUser(
            _make_profile(expected_outcomes=("good", "bad"))
        )
        score = user.evaluate_response("The outcome is good")
        assert score == pytest.approx(0.5)

    def test_empty_expected_outcomes_returns_one(self) -> None:
        user = SimulatedUser(_make_profile(expected_outcomes=()))
        score = user.evaluate_response("any response here")
        assert score == pytest.approx(1.0)

    def test_case_insensitive_matching(self) -> None:
        user = SimulatedUser(
            _make_profile(expected_outcomes=("SUCCESS",))
        )
        score = user.evaluate_response("The operation was a success")
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SimulatedUser.average_satisfaction()
# ---------------------------------------------------------------------------


class TestSimulatedUserAverageSatisfaction:
    def test_no_evaluations_returns_zero(self) -> None:
        user = SimulatedUser(_make_profile())
        assert user.average_satisfaction() == 0.0

    def test_single_evaluation_average(self) -> None:
        user = SimulatedUser(
            _make_profile(expected_outcomes=("hello",))
        )
        user.evaluate_response("hello there")
        assert user.average_satisfaction() == pytest.approx(1.0)

    def test_multiple_evaluations_averaged(self) -> None:
        user = SimulatedUser(
            _make_profile(expected_outcomes=("yes",))
        )
        user.evaluate_response("yes")  # 1.0
        user.evaluate_response("no")   # 0.0
        assert user.average_satisfaction() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SimulatedUser.reset()
# ---------------------------------------------------------------------------


class TestSimulatedUserReset:
    def test_reset_restores_cursor(self) -> None:
        user = SimulatedUser(_make_profile(messages=("A", "B")))
        user.next_message()
        user.next_message()
        user.reset()
        assert user.messages_sent == 0
        assert user.next_message() == "A"

    def test_reset_clears_satisfaction_scores(self) -> None:
        user = SimulatedUser(_make_profile(expected_outcomes=()))
        user.evaluate_response("hello")
        user.reset()
        assert user.average_satisfaction() == 0.0

    def test_repr_contains_name_and_persona(self) -> None:
        user = SimulatedUser(_make_profile(name="carol", persona="confused"))
        representation = repr(user)
        assert "carol" in representation
        assert "confused" in representation
