"""Unit tests for simulation backend stubs.

Covers:
- GazeboBackend: all methods raise NotImplementedError, __repr__
- PyBulletBackend: ImportError when pybullet absent, __repr__, all stubs
- backends/__init__.py: GazeboBackend always exported, PyBulletBackend optional
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent_sim_bridge.backends.gazebo import GazeboBackend


# ---------------------------------------------------------------------------
# GazeboBackend
# ---------------------------------------------------------------------------


class TestGazeboBackend:
    """GazeboBackend is a pure stub — every method must raise NotImplementedError."""

    def setup_method(self) -> None:
        self.backend = GazeboBackend()

    def test_repr(self) -> None:
        result = repr(self.backend)
        assert "GazeboBackend" in result
        assert "stub" in result

    def test_backend_reset_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="GazeboBackend is a stub"):
            self.backend.backend_reset(seed=None, options=None)

    def test_backend_step_raises(self) -> None:
        action = np.zeros((2,), dtype=np.float32)
        with pytest.raises(NotImplementedError):
            self.backend.backend_step(action)

    def test_backend_observe_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            self.backend.backend_observe()

    def test_backend_act_raises(self) -> None:
        action = np.zeros((2,), dtype=np.float32)
        with pytest.raises(NotImplementedError):
            self.backend.backend_act(action)

    def test_backend_close_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            self.backend.backend_close()

    def test_backend_state_space_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = self.backend.backend_state_space

    def test_backend_action_space_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = self.backend.backend_action_space

    def test_is_subclass_of_sim_backend(self) -> None:
        from agent_sim_bridge.environment.sim_env import SimBackend

        # SimBackend is a Protocol with non-method members (properties), which
        # prevents both isinstance() and issubclass() checks in Python 3.11.
        # Verify inheritance directly via the MRO instead.
        assert SimBackend in GazeboBackend.__mro__


# ---------------------------------------------------------------------------
# PyBulletBackend — pybullet NOT installed
# ---------------------------------------------------------------------------


class TestPyBulletBackendNoInstall:
    """When pybullet is absent, constructing PyBulletBackend must raise ImportError."""

    def test_import_error_when_pybullet_missing(self) -> None:
        # Simulate pybullet not being installed by patching the module-level flag.
        from agent_sim_bridge.backends import pybullet as pb_module

        original = pb_module._PYBULLET_AVAILABLE
        try:
            pb_module._PYBULLET_AVAILABLE = False
            with pytest.raises(ImportError, match="pip install pybullet"):
                from agent_sim_bridge.backends.pybullet import PyBulletBackend

                PyBulletBackend()
        finally:
            pb_module._PYBULLET_AVAILABLE = original


# ---------------------------------------------------------------------------
# PyBulletBackend — pybullet IS installed (mocked)
# ---------------------------------------------------------------------------


class TestPyBulletBackendWithMockedInstall:
    """With pybullet mocked as installed, PyBulletBackend stubs must behave correctly."""

    def setup_method(self) -> None:
        from agent_sim_bridge.backends import pybullet as pb_module

        self._original = pb_module._PYBULLET_AVAILABLE
        pb_module._PYBULLET_AVAILABLE = True
        from agent_sim_bridge.backends.pybullet import PyBulletBackend

        self.backend = PyBulletBackend(use_gui=False)
        self.pb_module = pb_module

    def teardown_method(self) -> None:
        self.pb_module._PYBULLET_AVAILABLE = self._original

    def test_repr_contains_use_gui(self) -> None:
        result = repr(self.backend)
        assert "PyBulletBackend" in result
        assert "False" in result

    def test_repr_gui_true(self) -> None:
        from agent_sim_bridge.backends.pybullet import PyBulletBackend

        backend_gui = PyBulletBackend(use_gui=True)
        result = repr(backend_gui)
        assert "True" in result

    def test_backend_reset_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="PyBulletBackend is a stub"):
            self.backend.backend_reset(seed=0, options=None)

    def test_backend_step_raises(self) -> None:
        action = np.zeros((3,), dtype=np.float32)
        with pytest.raises(NotImplementedError):
            self.backend.backend_step(action)

    def test_backend_observe_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            self.backend.backend_observe()

    def test_backend_act_raises(self) -> None:
        action = np.zeros((3,), dtype=np.float32)
        with pytest.raises(NotImplementedError):
            self.backend.backend_act(action)

    def test_backend_close_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            self.backend.backend_close()

    def test_backend_state_space_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = self.backend.backend_state_space

    def test_backend_action_space_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = self.backend.backend_action_space


# ---------------------------------------------------------------------------
# backends/__init__.py exports
# ---------------------------------------------------------------------------


class TestBackendsInit:
    def test_gazebo_backend_in_all(self) -> None:
        import agent_sim_bridge.backends as backends_pkg

        assert "GazeboBackend" in backends_pkg.__all__

    def test_gazebo_backend_importable(self) -> None:
        from agent_sim_bridge.backends import GazeboBackend as GB

        assert GB is GazeboBackend
