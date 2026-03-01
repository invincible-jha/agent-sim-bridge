# agent-sim-bridge

Simulation-to-reality bridge for AI agents with environment adapters

[![CI](https://github.com/aumos-ai/agent-sim-bridge/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-sim-bridge/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-sim-bridge.svg)](https://pypi.org/project/agent-sim-bridge/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-sim-bridge.svg)](https://pypi.org/project/agent-sim-bridge/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- `TransferBridge` applies a `CalibrationProfile` (per-dimension linear scale + offset + noise model) to translate observations and actions between simulator coordinates and real-world distributions
- `Calibrator` fits calibration profiles by regression on paired sim/real observation datasets, with support for iterative recalibration as new data arrives
- `DomainRandomization` perturbs simulation parameters (friction, mass, sensor noise, lighting) across configurable ranges to produce policies that transfer without fine-tuning
- Simulation scenario recorder and replay engine lets you capture real-world trajectories, replay them in simulation, and measure the fidelity gap
- `SafetyMonitor` enforces physical and policy constraints during both simulated and real execution, with configurable boundary definitions and emergency stop hooks
- Sensor fusion module combines readings from multiple simulated or real sensors with configurable noise injection and covariance-weighted fusion
- Backend adapters for Gazebo and PyBullet expose a uniform `SimulationEnvironment` interface so agents can be tested against either simulator without code changes

## Current Limitations

> **Transparency note**: We list known limitations to help you evaluate fit.

- **Fidelity**: Text-based scenarios only. No physics engine integration.
- **Validation**: No sim-to-real transfer validation harness yet.
- **Environments**: Mock environments — no hardware-in-the-loop support.

## Quick Start

Install from PyPI:

```bash
pip install agent-sim-bridge
```

Verify the installation:

```bash
agent-sim-bridge version
```

Basic usage:

```python
import agent_sim_bridge

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

For production deployments requiring SLA-backed support and advanced
integrations, contact the maintainers or see the commercial extensions documentation.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
