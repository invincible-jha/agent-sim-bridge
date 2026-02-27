"""CLI entry point for agent-sim-bridge.

Invoked as::

    agent-sim-bridge [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_sim_bridge.cli.main

Available commands
------------------
* ``sim run``          — execute a scenario episode in a named simulation environment
* ``sim record``       — record a trajectory to disk
* ``sim replay``       — replay a recorded trajectory and report divergence
* ``sim calibrate``    — run calibration from a paired sim/real data file
* ``sim gap-analysis`` — measure and report the sim-to-real gap
* ``sim safety-check`` — validate an action vector against a constraint file
* ``version``          — show detailed version information
* ``plugins``          — list registered plugins
"""
from __future__ import annotations

import json
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Set the log level.",
    show_default=True,
)
def cli(log_level: str) -> None:
    """Simulation-to-reality bridge for AI agents with environment adapters."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s %(name)s — %(message)s",
    )


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_sim_bridge import __version__

    console.print(f"[bold]agent-sim-bridge[/bold] v{__version__}")
    console.print(f"Python {sys.version}")


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered plugins loaded from entry-points."""
    console.print("[bold]Registered plugins:[/bold]")
    console.print(
        "  (No plugins registered. Install a plugin package to see entries here.)"
    )


# ---------------------------------------------------------------------------
# sim group
# ---------------------------------------------------------------------------


@cli.group(name="sim")
def sim_group() -> None:
    """Simulation execution and analysis commands."""


# ---------------------------------------------------------------------------
# sim run
# ---------------------------------------------------------------------------


@sim_group.command(name="run")
@click.option("--steps", default=100, show_default=True, help="Maximum episode steps.")
@click.option("--seed", default=None, type=int, help="RNG seed for reproducibility.")
@click.option(
    "--timeout",
    default=60.0,
    show_default=True,
    type=float,
    help="Wall-clock timeout in seconds.",
)
@click.option(
    "--record",
    "record_path",
    default=None,
    type=click.Path(),
    help="Save trajectory to this path (.npz).",
)
def sim_run(
    steps: int,
    seed: int | None,
    timeout: float,
    record_path: str | None,
) -> None:
    """Execute a simulation episode using a zero-action baseline policy.

    This command demonstrates the sandbox execution loop.  Replace the
    zero-action policy with a real agent by extending this command or
    calling :class:`~agent_sim_bridge.simulation.SimulationSandbox` directly.
    """
    from agent_sim_bridge.simulation.sandbox import SimulationSandbox

    console.print(
        f"[bold cyan]sim run[/bold cyan] — "
        f"steps={steps}, seed={seed}, timeout={timeout}s"
    )
    console.print(
        "[yellow]Note:[/yellow] No environment backend is configured. "
        "This command demonstrates the CLI surface only.\n"
        "Wire up a real backend by subclassing SimulationEnvironment."
    )

    # Show what a result would look like.
    console.print("\n[bold]Configuration[/bold]")
    table = Table(show_header=False, box=None)
    table.add_row("Max steps", str(steps))
    table.add_row("Seed", str(seed))
    table.add_row("Timeout (s)", str(timeout))
    table.add_row("Record path", str(record_path) if record_path else "(disabled)")
    console.print(table)


# ---------------------------------------------------------------------------
# sim record
# ---------------------------------------------------------------------------


@sim_group.command(name="record")
@click.argument("output_path", type=click.Path())
@click.option("--steps", default=200, show_default=True, help="Steps to record.")
@click.option("--seed", default=None, type=int, help="RNG seed.")
def sim_record(output_path: str, steps: int, seed: int | None) -> None:
    """Record an episode trajectory to OUTPUT_PATH (.npz).

    The trajectory is captured using
    :class:`~agent_sim_bridge.simulation.TrajectoryRecorder` and saved
    as a compressed numpy archive.  Replay the trajectory later with
    ``sim replay``.
    """
    from agent_sim_bridge.simulation.recorder import TrajectoryRecorder

    console.print(
        f"[bold cyan]sim record[/bold cyan] — output={output_path!r}, "
        f"steps={steps}, seed={seed}"
    )
    console.print(
        "[yellow]Note:[/yellow] No environment backend is configured. "
        "This command demonstrates the CLI surface only.\n"
        "Wire up a real backend to produce actual trajectory data."
    )

    # Demonstrate recorder API.
    recorder = TrajectoryRecorder(max_steps=steps)
    console.print(
        f"\nTrajectoryRecorder ready (max_steps={steps}). "
        f"Would save to: [bold]{output_path}[/bold]"
    )
    console.print(f"Recorder state: {recorder!r}")


# ---------------------------------------------------------------------------
# sim replay
# ---------------------------------------------------------------------------


@sim_group.command(name="replay")
@click.argument("trajectory_path", type=click.Path(exists=True))
@click.option("--seed", default=None, type=int, help="Reset seed for the replay environment.")
@click.option(
    "--stop-on-termination/--no-stop-on-termination",
    default=True,
    show_default=True,
    help="Stop early if the environment terminates.",
)
def sim_replay(
    trajectory_path: str,
    seed: int | None,
    stop_on_termination: bool,
) -> None:
    """Replay a recorded trajectory from TRAJECTORY_PATH and report divergence.

    Loads the trajectory, feeds actions back through the simulation, and
    computes the observation MSE and reward correlation against the original.
    """
    from agent_sim_bridge.simulation.recorder import TrajectoryRecorder

    console.print(
        f"[bold cyan]sim replay[/bold cyan] — path={trajectory_path!r}, seed={seed}"
    )

    try:
        recorder = TrajectoryRecorder.load(trajectory_path)
    except Exception as exc:
        console.print(f"[red]Error loading trajectory:[/red] {exc}")
        raise SystemExit(1) from exc

    console.print(f"\nLoaded trajectory: {recorder!r}")
    console.print(
        "[yellow]Note:[/yellow] No environment backend is configured. "
        "Connect a SimulationEnvironment and TrajectoryReplay to run the full replay."
    )

    # Show trajectory summary.
    table = Table(title="Trajectory Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Steps loaded", str(len(recorder)))
    table.add_row("Stop on termination", str(stop_on_termination))
    table.add_row("Replay seed", str(seed))
    console.print(table)


# ---------------------------------------------------------------------------
# sim calibrate
# ---------------------------------------------------------------------------


@sim_group.command(name="calibrate")
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Save calibration profile as JSON to this path.",
)
def sim_calibrate(data_path: str, output: str | None) -> None:
    """Fit a CalibrationProfile from paired sim/real observations in DATA_PATH.

    DATA_PATH must be a JSON file with keys ``"sim_obs"`` and ``"real_obs"``,
    each a list of observation vectors (list-of-lists of floats).

    Example JSON format::

        {
            "sim_obs": [[1.0, 2.0], [1.1, 2.1]],
            "real_obs": [[1.05, 1.98], [1.15, 2.08]]
        }
    """
    import pathlib

    from agent_sim_bridge.transfer.calibration import Calibrator

    console.print(f"[bold cyan]sim calibrate[/bold cyan] — data={data_path!r}")

    try:
        raw = pathlib.Path(data_path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        console.print(f"[red]Error reading data file:[/red] {exc}")
        raise SystemExit(1) from exc

    sim_obs: list[list[float]] = data.get("sim_obs", [])
    real_obs: list[list[float]] = data.get("real_obs", [])

    if not sim_obs or not real_obs:
        console.print(
            "[red]Data file must contain non-empty 'sim_obs' and 'real_obs' keys.[/red]"
        )
        raise SystemExit(1)

    calibrator = Calibrator()
    try:
        profile = calibrator.calibrate(sim_obs, real_obs)
    except Exception as exc:
        console.print(f"[red]Calibration failed:[/red] {exc}")
        raise SystemExit(1) from exc

    mae = calibrator.evaluate_calibration(sim_obs, real_obs)

    table = Table(title="Calibration Results", show_header=True)
    table.add_column("Dimension", style="bold")
    table.add_column("Scale Factor")
    table.add_column("Offset")
    for i, (scale, offset) in enumerate(zip(profile.scale_factors, profile.offsets)):
        table.add_row(str(i), f"{scale:.6f}", f"{offset:.6f}")
    console.print(table)
    console.print(f"\nMAE (calibration error): [bold]{mae:.6f}[/bold]")
    console.print(f"Samples: {len(sim_obs)}, Dimensions: {profile.n_dims}")

    if output:
        import dataclasses

        pathlib.Path(output).write_text(
            json.dumps(dataclasses.asdict(profile), indent=2), encoding="utf-8"
        )
        console.print(f"\nProfile saved to: [bold]{output}[/bold]")


# ---------------------------------------------------------------------------
# sim gap-analysis
# ---------------------------------------------------------------------------


@sim_group.command(name="gap-analysis")
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Save GapReport as JSON to this path.",
)
def sim_gap_analysis(data_path: str, output: str | None) -> None:
    """Measure and report the sim-to-real gap from paired trajectory data.

    DATA_PATH must be a JSON file with keys ``"sim_obs"``, ``"real_obs"``,
    and optionally ``"sim_rewards"`` and ``"real_rewards"``.

    Example JSON format::

        {
            "sim_obs":     [[1.0, 2.0], [1.1, 2.1]],
            "real_obs":    [[1.05, 1.98], [1.15, 2.08]],
            "sim_rewards": [0.5, 0.6],
            "real_rewards":[0.48, 0.57]
        }
    """
    import dataclasses
    import pathlib

    from agent_sim_bridge.metrics.gap import SimRealGap

    console.print(f"[bold cyan]sim gap-analysis[/bold cyan] — data={data_path!r}")

    try:
        raw = pathlib.Path(data_path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        console.print(f"[red]Error reading data file:[/red] {exc}")
        raise SystemExit(1) from exc

    sim_obs: list[list[float]] = data.get("sim_obs", [])
    real_obs: list[list[float]] = data.get("real_obs", [])
    sim_rewards: list[float] | None = data.get("sim_rewards")
    real_rewards: list[float] | None = data.get("real_rewards")

    if not sim_obs or not real_obs:
        console.print(
            "[red]Data file must contain non-empty 'sim_obs' and 'real_obs' keys.[/red]"
        )
        raise SystemExit(1)

    gap_metric = SimRealGap()
    try:
        report = gap_metric.measure_gap(sim_obs, real_obs, sim_rewards, real_rewards)
    except Exception as exc:
        console.print(f"[red]Gap analysis failed:[/red] {exc}")
        raise SystemExit(1) from exc

    table = Table(title="Sim-to-Real Gap Report", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for metric_name, value in report.summary().items():
        table.add_row(metric_name, f"{value:.6f}" if isinstance(value, float) else str(value))
    console.print(table)

    if output:
        pathlib.Path(output).write_text(
            json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8"
        )
        console.print(f"\nReport saved to: [bold]{output}[/bold]")


# ---------------------------------------------------------------------------
# sim safety-check
# ---------------------------------------------------------------------------


@sim_group.command(name="safety-check")
@click.argument("action_json")
@click.option(
    "--constraints",
    "constraints_path",
    default=None,
    type=click.Path(exists=True),
    help="JSON file defining safety constraints.",
)
def sim_safety_check(action_json: str, constraints_path: str | None) -> None:
    """Validate ACTION_JSON against safety constraints.

    ACTION_JSON is a JSON-encoded list of floats representing an action
    vector, e.g.::

        '[0.5, -0.3, 1.2]'

    CONSTRAINTS_PATH (optional) is a JSON file containing a list of
    constraint definitions, each with keys: ``name``, ``constraint_type``,
    ``dimension``, ``min_value``, ``max_value``, ``severity``.

    Example constraints file::

        [
            {
                "name": "joint_0_range",
                "constraint_type": "range",
                "dimension": 0,
                "min_value": -1.0,
                "max_value": 1.0,
                "severity": "error"
            }
        ]
    """
    import pathlib

    from agent_sim_bridge.safety.constraints import (
        ConstraintType,
        SafetyChecker,
        SafetyConstraint,
        ViolationSeverity,
    )

    console.print(f"[bold cyan]sim safety-check[/bold cyan]")

    try:
        action: list[float] = json.loads(action_json)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid ACTION_JSON:[/red] {exc}")
        raise SystemExit(1) from exc

    if not isinstance(action, list) or not all(isinstance(v, (int, float)) for v in action):
        console.print("[red]ACTION_JSON must be a JSON array of numbers.[/red]")
        raise SystemExit(1)

    action_floats = [float(v) for v in action]

    # Load or use default constraints.
    constraints: list[SafetyConstraint] = []
    if constraints_path:
        try:
            raw = pathlib.Path(constraints_path).read_text(encoding="utf-8")
            constraint_data: list[dict[str, object]] = json.loads(raw)
        except Exception as exc:
            console.print(f"[red]Error reading constraints file:[/red] {exc}")
            raise SystemExit(1) from exc

        for item in constraint_data:
            constraints.append(
                SafetyConstraint(
                    name=str(item.get("name", "unnamed")),
                    constraint_type=ConstraintType(
                        item.get("constraint_type", "range")
                    ),
                    dimension=int(item.get("dimension", 0)),  # type: ignore[arg-type]
                    min_value=float(item.get("min_value", float("-inf"))),  # type: ignore[arg-type]
                    max_value=float(item.get("max_value", float("inf"))),  # type: ignore[arg-type]
                    severity=ViolationSeverity(item.get("severity", "error")),
                )
            )
    else:
        # Default: range check [-1, 1] for each dimension.
        for dim in range(len(action_floats)):
            constraints.append(
                SafetyConstraint(
                    name=f"dim_{dim}_range",
                    dimension=dim,
                    min_value=-1.0,
                    max_value=1.0,
                )
            )
        console.print(
            "[yellow]No constraints file provided.[/yellow] "
            "Using default range [-1, 1] for each dimension."
        )

    checker = SafetyChecker()
    violations = checker.check(action_floats, constraints)

    console.print(f"\nAction: {action_floats}")
    console.print(f"Constraints checked: {len(constraints)}")

    if violations:
        console.print(f"\n[bold red]Violations found: {len(violations)}[/bold red]")
        table = Table(title="Safety Violations", show_header=True)
        table.add_column("Constraint", style="bold")
        table.add_column("Severity")
        table.add_column("Value")
        table.add_column("Message")
        for violation in violations:
            table.add_row(
                violation.constraint_name,
                violation.severity.value,
                f"{violation.actual_value:.6f}",
                violation.message,
            )
        console.print(table)
        raise SystemExit(1)
    else:
        console.print("\n[bold green]All safety checks passed.[/bold green]")


# ---------------------------------------------------------------------------
# gap group
# ---------------------------------------------------------------------------


@cli.group(name="gap")
def gap_group() -> None:
    """Distribution-based sim-to-real gap estimation commands."""


# ---------------------------------------------------------------------------
# gap estimate
# ---------------------------------------------------------------------------


@gap_group.command(name="estimate")
@click.option(
    "--sim-data",
    "sim_data_path",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with simulation distribution data.",
)
@click.option(
    "--real-data",
    "real_data_path",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with real-world distribution data.",
)
@click.option(
    "--metrics",
    "metrics_str",
    default=None,
    help=(
        "Comma-separated list of metrics to compute: "
        "kl,wasserstein,mmd,jsd.  Defaults to all four."
    ),
)
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "markdown"], case_sensitive=False),
    show_default=True,
    help="Output format for the gap report.",
)
def gap_estimate(
    sim_data_path: str,
    real_data_path: str,
    metrics_str: str | None,
    output_format: str,
) -> None:
    """Estimate the sim-to-real gap from distribution data files.

    Both data files must be JSON with the schema::

        {"dimensions": {"dim_name": {"sim": [values], "real": [values]}}}

    Example::

        agent-sim-bridge gap estimate \\
            --sim-data sim.json --real-data real.json \\
            --metrics kl,wasserstein --format markdown
    """
    import pathlib

    from agent_sim_bridge.gap.estimator import GapDimension, GapEstimator, GapMetric
    from agent_sim_bridge.gap.report import GapReporter

    _METRIC_MAP: dict[str, GapMetric] = {
        "kl": GapMetric.KL_DIVERGENCE,
        "wasserstein": GapMetric.WASSERSTEIN,
        "mmd": GapMetric.MMD,
        "jsd": GapMetric.JENSEN_SHANNON,
    }

    # Load data files.
    try:
        sim_raw = json.loads(pathlib.Path(sim_data_path).read_text(encoding="utf-8"))
        real_raw = json.loads(pathlib.Path(real_data_path).read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Error reading data files:[/red] {exc}")
        raise SystemExit(1) from exc

    sim_dimensions: dict[str, object] = sim_raw.get("dimensions", {})
    real_dimensions: dict[str, object] = real_raw.get("dimensions", {})

    if not sim_dimensions or not real_dimensions:
        console.print(
            "[red]Both data files must contain a non-empty 'dimensions' key.[/red]"
        )
        raise SystemExit(1)

    # Resolve metrics.
    selected_metrics: list[GapMetric] | None = None
    if metrics_str:
        selected_metrics = []
        for token in metrics_str.split(","):
            token = token.strip().lower()
            if token not in _METRIC_MAP:
                console.print(
                    f"[red]Unknown metric {token!r}. "
                    f"Choose from: {', '.join(_METRIC_MAP)}.[/red]"
                )
                raise SystemExit(1)
            selected_metrics.append(_METRIC_MAP[token])

    # Build dimensions.
    common_names = set(sim_dimensions) & set(real_dimensions)
    if not common_names:
        console.print(
            "[red]No matching dimension names found between sim and real data files.[/red]"
        )
        raise SystemExit(1)

    dimensions: list[GapDimension] = []
    for dim_name in sorted(common_names):
        sim_entry = sim_dimensions[dim_name]
        real_entry = real_dimensions[dim_name]
        try:
            sim_values: list[float] = list(sim_entry["sim"])   # type: ignore[index]
            real_values: list[float] = list(real_entry["real"])  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            console.print(
                f"[red]Dimension {dim_name!r} must have 'sim' and 'real' keys.[/red] {exc}"
            )
            raise SystemExit(1) from exc
        dimensions.append(
            GapDimension(
                name=dim_name,
                sim_distribution=sim_values,
                real_distribution=real_values,
            )
        )

    # Run estimation.
    estimator = GapEstimator(metrics=selected_metrics)
    try:
        results = estimator.estimate_all(dimensions)
    except Exception as exc:
        console.print(f"[red]Gap estimation failed:[/red] {exc}")
        raise SystemExit(1) from exc

    reporter = GapReporter()
    report = reporter.generate_report(results, dimensions)

    if output_format == "json":
        console.print(reporter.format_json(report))
    elif output_format == "markdown":
        console.print(reporter.format_markdown(report))
    else:
        console.print(reporter.format_text(report))


# ---------------------------------------------------------------------------
# gap compare
# ---------------------------------------------------------------------------


@gap_group.command(name="compare")
@click.option(
    "--sim-data",
    "sim_data_path",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with simulation distribution data.",
)
@click.option(
    "--real-data",
    "real_data_path",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with real-world distribution data.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Path to write the JSON gap report.",
)
def gap_compare(
    sim_data_path: str,
    real_data_path: str,
    output_path: str,
) -> None:
    """Compare sim and real distributions and save a full JSON report.

    Both data files must be JSON with the schema::

        {"dimensions": {"dim_name": {"sim": [values], "real": [values]}}}

    The report is written to OUTPUT as a JSON file.

    Example::

        agent-sim-bridge gap compare \\
            --sim-data sim.json --real-data real.json \\
            --output report.json
    """
    import pathlib

    from agent_sim_bridge.gap.estimator import GapDimension, GapEstimator
    from agent_sim_bridge.gap.report import GapReporter

    # Load data files.
    try:
        sim_raw = json.loads(pathlib.Path(sim_data_path).read_text(encoding="utf-8"))
        real_raw = json.loads(pathlib.Path(real_data_path).read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Error reading data files:[/red] {exc}")
        raise SystemExit(1) from exc

    sim_dimensions: dict[str, object] = sim_raw.get("dimensions", {})
    real_dimensions: dict[str, object] = real_raw.get("dimensions", {})

    if not sim_dimensions or not real_dimensions:
        console.print(
            "[red]Both data files must contain a non-empty 'dimensions' key.[/red]"
        )
        raise SystemExit(1)

    common_names = set(sim_dimensions) & set(real_dimensions)
    if not common_names:
        console.print(
            "[red]No matching dimension names found between sim and real data files.[/red]"
        )
        raise SystemExit(1)

    dimensions: list[GapDimension] = []
    for dim_name in sorted(common_names):
        sim_entry = sim_dimensions[dim_name]
        real_entry = real_dimensions[dim_name]
        try:
            sim_values: list[float] = list(sim_entry["sim"])   # type: ignore[index]
            real_values: list[float] = list(real_entry["real"])  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            console.print(
                f"[red]Dimension {dim_name!r} missing 'sim' or 'real' key.[/red] {exc}"
            )
            raise SystemExit(1) from exc
        dimensions.append(
            GapDimension(
                name=dim_name,
                sim_distribution=sim_values,
                real_distribution=real_values,
            )
        )

    estimator = GapEstimator()
    try:
        results = estimator.estimate_all(dimensions)
    except Exception as exc:
        console.print(f"[red]Gap estimation failed:[/red] {exc}")
        raise SystemExit(1) from exc

    reporter = GapReporter()
    report = reporter.generate_report(results, dimensions)
    json_output = reporter.format_json(report)

    try:
        pathlib.Path(output_path).write_text(json_output, encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]Error writing report:[/red] {exc}")
        raise SystemExit(1) from exc

    console.print(f"[bold green]Gap report saved to:[/bold green] {output_path}")
    console.print(f"Overall gap score: [bold]{report.overall_score:.4f}[/bold]")
    console.print(f"Summary: {report.summary}")


if __name__ == "__main__":
    cli()
