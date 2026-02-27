# agent-sim-bridge

Simulation-to-Reality Bridge — environment adapters, transfer learning, domain randomization, and gap estimation for AI agents.

[![CI](https://github.com/invincible-jha/agent-sim-bridge/actions/workflows/ci.yaml/badge.svg)](https://github.com/invincible-jha/agent-sim-bridge/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-sim-bridge.svg)](https://pypi.org/project/agent-sim-bridge/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-sim-bridge.svg)](https://pypi.org/project/agent-sim-bridge/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Installation

```bash
pip install agent-sim-bridge
```

Verify the installation:

```bash
agent-sim-bridge version
```

---

## Quick Start

```python
import agent_sim_bridge

# See examples/01_quickstart.py for a working example
```

---

## Key Features

- **TransferBridge** applies a `CalibrationProfile` (per-dimension linear scale + offset + noise model) to translate observations and actions between simulator coordinates and real-world distributions
- **Calibrator** fits calibration profiles by regression on paired sim/real observation datasets, with support for iterative recalibration as new data arrives
- **DomainRandomization** perturbs simulation parameters (friction, mass, sensor noise, lighting) across configurable ranges to produce policies that transfer without fine-tuning
- **Scenario recorder and replay engine** lets you capture real-world trajectories, replay them in simulation, and measure the fidelity gap
- **SafetyMonitor** enforces physical and policy constraints during both simulated and real execution, with configurable boundary definitions and emergency stop hooks
- **Sensor fusion module** combines readings from multiple simulated or real sensors with configurable noise injection and covariance-weighted fusion
- **Backend adapters for Gazebo and PyBullet** expose a uniform `SimulationEnvironment` interface so agents can be tested against either simulator without code changes

---

## Links

- [GitHub Repository](https://github.com/invincible-jha/agent-sim-bridge)
- [PyPI Package](https://pypi.org/project/agent-sim-bridge/)
- [Architecture](architecture.md)
- [Changelog](https://github.com/invincible-jha/agent-sim-bridge/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/invincible-jha/agent-sim-bridge/blob/main/CONTRIBUTING.md)

---

> Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.
