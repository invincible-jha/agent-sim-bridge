"""GazeboBackend — stub for the Gazebo simulation backend.

Gazebo (https://gazebosim.org) is a popular robotics simulator used in ROS
ecosystems.  Full integration requires the ``gz-python`` or ``pygazebo``
bindings which are not included in the base package.

All methods raise :class:`NotImplementedError` until a real implementation
is provided.  Install the optional ``[gazebo]`` extra and override this
class, or register a concrete subclass via the plugin registry.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from agent_sim_bridge.environment.base import SpaceSpec
from agent_sim_bridge.environment.sim_env import SimBackend

logger = logging.getLogger(__name__)

_STUB_MESSAGE = (
    "GazeboBackend is a stub.  Install the optional Gazebo integration "
    "package and provide a concrete implementation, or register a "
    "subclass via the plugin registry."
)


class GazeboBackend(SimBackend):
    """Stub :class:`~agent_sim_bridge.environment.sim_env.SimBackend` for Gazebo.

    All methods raise :class:`NotImplementedError`.  This class exists to:

    1. Define the expected interface so downstream packages can subclass it.
    2. Provide a named entry-point in the plugin registry.
    3. Be listed in ``__all__`` for IDE discoverability.

    To implement a real integration, subclass this and override all methods.
    """

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
        return "GazeboBackend(status='stub — not implemented')"
