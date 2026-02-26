"""Sim-to-real transfer utilities.

Provides linear calibration, domain randomization, and the bridge that
translates observations and actions between simulation and reality domains.
"""
from __future__ import annotations

from agent_sim_bridge.transfer.bridge import CalibrationProfile, TransferBridge
from agent_sim_bridge.transfer.calibration import Calibrator
from agent_sim_bridge.transfer.domain_randomization import (
    DistributionType,
    DomainRandomizer,
    RandomizationConfig,
)

__all__ = [
    "CalibrationProfile",
    "TransferBridge",
    "Calibrator",
    "RandomizationConfig",
    "DistributionType",
    "DomainRandomizer",
]
