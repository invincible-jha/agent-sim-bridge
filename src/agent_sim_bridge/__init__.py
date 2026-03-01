"""agent-sim-bridge — Simulation-to-reality bridge for AI agents with environment adapters.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Quick-start example
-------------------
>>> import agent_sim_bridge as asb
>>> asb.__version__
'0.1.0'

Subpackages
-----------
environment:
    Abstract base, simulation, real-world, and adapter environments.
simulation:
    Trajectory recording, replay, sandboxed execution, and scenario management.
transfer:
    Sim-to-real calibration, linear bridging, and domain randomization.
safety:
    Constraint definitions, real-time monitoring, and boundary checking.
sensors:
    Sensor ABC, readings, fusion strategies, and noise models.
skills:
    Reusable agent skill primitives, composer, and library.
metrics:
    Sim-to-real gap analysis and performance metric tracking.
backends:
    Stub backends for Gazebo and PyBullet.
plugins:
    Decorator-based plugin registration via entry-points.
validation:
    Sim-to-real validation harness with standard scenarios and fidelity reports.
"""
from __future__ import annotations

__version__: str = "0.1.0"

from agent_sim_bridge.convenience import (
    Simulator,
    quick_gap_analysis,
    quick_recorder,
    quick_safety_monitor,
    quick_sandbox,
    quick_skill_library,
    quick_transfer_bridge,
)

# -- Environment ----------------------------------------------------------
from agent_sim_bridge.environment import (
    Environment,
    EnvironmentAdapter,
    EnvironmentInfo,
    RealityEnvironment,
    SimulationEnvironment,
    StepResult,
)

# -- Simulation -----------------------------------------------------------
from agent_sim_bridge.simulation import (
    ExecutionResult,
    Scenario,
    ScenarioManager,
    SimulationSandbox,
    TrajectoryRecorder,
    TrajectoryReplay,
    TrajectoryStep,
)

# -- Transfer -------------------------------------------------------------
from agent_sim_bridge.transfer import (
    CalibrationProfile,
    Calibrator,
    DistributionType,
    DomainRandomizer,
    RandomizationConfig,
    TransferBridge,
)

# -- Safety ---------------------------------------------------------------
from agent_sim_bridge.safety import (
    BoundaryChecker,
    BoundaryDefinition,
    BoundaryViolation,
    ConstraintType,
    MonitoredStep,
    SafetyChecker,
    SafetyConstraint,
    SafetyMonitor,
    SafetyViolation,
    ViolationSeverity,
)

# -- Sensors --------------------------------------------------------------
from agent_sim_bridge.sensors import (
    CompositeNoise,
    FusionStrategy,
    GaussianNoise,
    NoiseModel,
    Sensor,
    SensorFusion,
    SensorReading,
    SensorType,
    UniformNoise,
)

# -- Skills ---------------------------------------------------------------
from agent_sim_bridge.skills import (
    CompositeSkill,
    Skill,
    SkillComposer,
    SkillLibrary,
    SkillNotFoundError,
    SkillResult,
    SkillStatus,
)

# -- Metrics --------------------------------------------------------------
from agent_sim_bridge.metrics import (
    GapReport,
    MetricRecord,
    PerformanceTracker,
    SimRealGap,
)

# -- Plugins --------------------------------------------------------------
from agent_sim_bridge.plugins import PluginRegistry

# -- Validation -----------------------------------------------------------
from agent_sim_bridge.validation import (
    Environment as ValidationEnvironment,
    EnvironmentInput,
    EnvironmentOutput,
    FidelityReport,
    MockEnvironment,
    ScenarioResult,
    STANDARD_SCENARIOS,
    ValidationHarness,
    ValidationScenario,
)

__all__: list[str] = [
    "__version__",
    # convenience
    "Simulator",
    "quick_gap_analysis",
    "quick_recorder",
    "quick_safety_monitor",
    "quick_sandbox",
    "quick_skill_library",
    "quick_transfer_bridge",
    # environment
    "Environment",
    "EnvironmentInfo",
    "StepResult",
    "SimulationEnvironment",
    "RealityEnvironment",
    "EnvironmentAdapter",
    # simulation
    "SimulationSandbox",
    "ExecutionResult",
    "TrajectoryRecorder",
    "TrajectoryStep",
    "TrajectoryReplay",
    "Scenario",
    "ScenarioManager",
    # transfer
    "CalibrationProfile",
    "TransferBridge",
    "Calibrator",
    "RandomizationConfig",
    "DistributionType",
    "DomainRandomizer",
    # safety
    "SafetyConstraint",
    "SafetyChecker",
    "SafetyViolation",
    "ConstraintType",
    "ViolationSeverity",
    "SafetyMonitor",
    "MonitoredStep",
    "BoundaryDefinition",
    "BoundaryChecker",
    "BoundaryViolation",
    # sensors
    "Sensor",
    "SensorReading",
    "SensorType",
    "SensorFusion",
    "FusionStrategy",
    "NoiseModel",
    "GaussianNoise",
    "UniformNoise",
    "CompositeNoise",
    # skills
    "Skill",
    "SkillResult",
    "SkillStatus",
    "SkillComposer",
    "CompositeSkill",
    "SkillLibrary",
    "SkillNotFoundError",
    # metrics
    "SimRealGap",
    "GapReport",
    "PerformanceTracker",
    "MetricRecord",
    # plugins
    "PluginRegistry",
    # validation
    "ValidationEnvironment",
    "EnvironmentInput",
    "EnvironmentOutput",
    "MockEnvironment",
    "ValidationScenario",
    "STANDARD_SCENARIOS",
    "ScenarioResult",
    "FidelityReport",
    "ValidationHarness",
]
