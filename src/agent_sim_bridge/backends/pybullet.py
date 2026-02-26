"""PyBulletBackend — stub for the PyBullet physics simulation backend.

PyBullet (https://pybullet.org) is a Python interface to the Bullet physics
engine, commonly used for robotics and locomotion research.

The import of ``pybullet`` is guarded so that the rest of the package can
be used without PyBullet installed.  Attempting to instantiate
:class:`PyBulletBackend` without PyBullet installed raises
:class:`ImportError` at construction time.

All abstract methods raise :class:`NotImplementedError` until a real
implementation is provided.

Optional install
----------------
::

    pip install pybullet

Then subclass :class:`PyBulletBackend` and implement each method.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from agent_sim_bridge.environment.base import SpaceSpec
from agent_sim_bridge.environment.sim_env import SimBackend

logger = logging.getLogger(__name__)

# Guarded import — pybullet is optional.
try:
    import pybullet as _pybullet  # noqa: F401

    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False

_STUB_MESSAGE = (
    "PyBulletBackend is a stub.  Install PyBullet (`pip install pybullet`) "
    "and provide a concrete implementation by subclassing PyBulletBackend."
)


class PyBulletBackend(SimBackend):
    """Stub :class:`~agent_sim_bridge.environment.sim_env.SimBackend` for PyBullet.

    Raises :class:`ImportError` at instantiation if ``pybullet`` is not
    installed, so the error is reported early and clearly.

    All interface methods raise :class:`NotImplementedError`.  Subclass
    and override them to provide a real integration.

    Parameters
    ----------
    use_gui:
        When True, the Bullet physics server starts in GUI mode.  Requires
        a display; use False for headless servers.
    """

    def __init__(self, use_gui: bool = False) -> None:
        if not _PYBULLET_AVAILABLE:
            raise ImportError(
                "PyBullet is not installed.  Install it with: pip install pybullet"
            )
        self._use_gui = use_gui
        logger.debug(
            "PyBulletBackend initialised (use_gui=%s). "
            "Note: this is a stub — override all abstract methods.",
            use_gui,
        )

    def backend_reset(
        self,
        seed: int | None,
        options: dict[str, object] | None,
    ) -> NDArray[np.float32]:
        raise NotImplementedError(_STUB_MESSAGE)

    def backend_step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        raise NotImplementedError(_STUB_MESSAGE)

    def backend_observe(self) -> NDArray[np.float32]:
        raise NotImplementedError(_STUB_MESSAGE)

    def backend_act(self, action: NDArray[np.float32]) -> None:
        raise NotImplementedError(_STUB_MESSAGE)

    def backend_close(self) -> None:
        raise NotImplementedError(_STUB_MESSAGE)

    @property
    def backend_state_space(self) -> SpaceSpec:
        raise NotImplementedError(_STUB_MESSAGE)

    @property
    def backend_action_space(self) -> SpaceSpec:
        raise NotImplementedError(_STUB_MESSAGE)

    def __repr__(self) -> str:
        return f"PyBulletBackend(use_gui={self._use_gui!r}, status='stub — not implemented')"
