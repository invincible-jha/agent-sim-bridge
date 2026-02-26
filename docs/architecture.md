# Architecture — agent-sim-bridge

## Overview

Simulation-to-reality bridge for AI agents with environment adapters

This document describes the high-level architecture of agent-sim-bridge
and the design decisions behind it.

## Component Map

```
agent-sim-bridge/
  src/agent_sim_bridge/
    core/        # Domain logic, models, protocols
    plugins/     # Plugin registry and base classes
    cli/         # Click CLI application
```

## Plugin System

agent-sim-bridge uses a decorator-based plugin registry backed by
``importlib.metadata`` entry-points. This allows third-party packages
(including the AgentPhysical enterprise edition) to extend the system
without modifying the core.

### Registration at import time

```python
from agent_sim_bridge.plugins.registry import PluginRegistry
from agent_sim_bridge.core import BaseProcessor  # example base class

processor_registry: PluginRegistry[BaseProcessor] = PluginRegistry(
    BaseProcessor, "processors"
)

@processor_registry.register("my-processor")
class MyProcessor(BaseProcessor):
    ...
```

### Registration via entry-points

Downstream packages declare plugins in ``pyproject.toml``:

```toml
[agent_sim_bridge.plugins]
my-processor = "my_package:MyProcessor"
```

Then load them at startup:

```python
processor_registry.load_entrypoints("agent_sim_bridge.plugins")
```

## Design Principles

- **Dependency injection**: services receive dependencies as constructor
  arguments rather than reaching for globals.
- **Pydantic v2 at boundaries**: all data entering or leaving the system
  is validated via Pydantic models.
- **Async-first**: I/O-bound operations use ``async``/``await``.
- **No hidden globals**: avoid module-level singletons that complicate
  testing and concurrent use.

## Extension Points

| Extension Point | Mechanism |
|----------------|-----------|
| Custom processors | ``PluginRegistry`` entry-points |
| Custom CLI commands | ``click`` group plugins |
| Configuration | Pydantic ``BaseSettings`` |

## Future Work

- [ ] Async streaming support
- [ ] OpenTelemetry tracing
- [ ] gRPC transport option
