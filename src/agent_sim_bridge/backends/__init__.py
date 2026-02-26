"""Backend stubs for simulation physics engines.

Provides stub implementations of
:class:`~agent_sim_bridge.environment.sim_env.SimBackend` that raise
:class:`NotImplementedError` until real integrations are supplied.

Available stubs
---------------
* :class:`~agent_sim_bridge.backends.gazebo.GazeboBackend` — Gazebo / gz-python.
* :class:`~agent_sim_bridge.backends.pybullet.PyBulletBackend` — PyBullet
  (import guarded; raises :class:`ImportError` if ``pybullet`` not installed).
"""
from __future__ import annotations

from agent_sim_bridge.backends.gazebo import GazeboBackend

__all__ = [
    "GazeboBackend",
]

# PyBulletBackend is optionally available — expose only when pybullet is installed.
try:
    from agent_sim_bridge.backends.pybullet import PyBulletBackend

    __all__ = [*__all__, "PyBulletBackend"]
except ImportError:
    pass
