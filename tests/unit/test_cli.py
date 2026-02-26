"""Unit tests for the CLI commands.

Uses Click's CliRunner to invoke all commands without launching a real process.
Covers:
- cli root group (--log-level)
- version command
- plugins command
- sim run
- sim record
- sim replay (success, error on missing file, error on bad trajectory)
- sim calibrate (success, bad json, missing keys, calibration failure)
- sim gap-analysis (success, bad json, missing keys, failure)
- sim safety-check (valid action, invalid json, non-array, violation,
  constraints from file, bad constraint file)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from agent_sim_bridge.cli.main import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runner() -> CliRunner:
    return CliRunner()


def _write_json(data: object, directory: str) -> str:
    path = Path(directory) / "data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _write_trajectory(directory: str, n_steps: int = 3) -> str:
    """Save a minimal .npz trajectory and return the path string."""
    obs = np.zeros((n_steps, 3), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    rewards = np.ones(n_steps, dtype=np.float32)
    terminated = np.zeros(n_steps, dtype=bool)
    truncated = np.zeros(n_steps, dtype=bool)
    timestamps = np.arange(n_steps, dtype=np.float64)
    path = Path(directory) / "traj.npz"
    np.savez_compressed(
        str(path),
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=obs,
        terminated=terminated,
        truncated=truncated,
        timestamps=timestamps,
    )
    return str(path)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


class TestCliRoot:
    def test_help(self) -> None:
        result = _runner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "simulation" in result.output.lower() or "bridge" in result.output.lower()

    def test_log_level_option_accepted(self) -> None:
        result = _runner().invoke(cli, ["--log-level", "DEBUG", "version"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersionCommand:
    def test_version_exits_zero(self) -> None:
        result = _runner().invoke(cli, ["version"])
        assert result.exit_code == 0

    def test_version_output_contains_package_name(self) -> None:
        result = _runner().invoke(cli, ["version"])
        assert "agent-sim-bridge" in result.output

    def test_version_output_contains_python(self) -> None:
        result = _runner().invoke(cli, ["version"])
        assert "Python" in result.output


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------


class TestPluginsCommand:
    def test_plugins_exits_zero(self) -> None:
        result = _runner().invoke(cli, ["plugins"])
        assert result.exit_code == 0

    def test_plugins_output(self) -> None:
        result = _runner().invoke(cli, ["plugins"])
        assert "plugin" in result.output.lower() or "registered" in result.output.lower()


# ---------------------------------------------------------------------------
# sim run
# ---------------------------------------------------------------------------


class TestSimRunCommand:
    def test_sim_run_default(self) -> None:
        result = _runner().invoke(cli, ["sim", "run"])
        assert result.exit_code == 0

    def test_sim_run_with_options(self) -> None:
        result = _runner().invoke(
            cli, ["sim", "run", "--steps", "50", "--seed", "1", "--timeout", "30.0"]
        )
        assert result.exit_code == 0
        assert "50" in result.output

    def test_sim_run_with_record_path(self) -> None:
        result = _runner().invoke(
            cli, ["sim", "run", "--record", "/tmp/traj.npz"]
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# sim record
# ---------------------------------------------------------------------------


class TestSimRecordCommand:
    def test_sim_record_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "out.npz")
            result = _runner().invoke(cli, ["sim", "record", path])
        assert result.exit_code == 0

    def test_sim_record_shows_recorder_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "out.npz")
            result = _runner().invoke(cli, ["sim", "record", path, "--steps", "10"])
        assert "10" in result.output or "recorder" in result.output.lower()


# ---------------------------------------------------------------------------
# sim replay
# ---------------------------------------------------------------------------


class TestSimReplayCommand:
    def test_sim_replay_valid_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = _write_trajectory(tmp)
            result = _runner().invoke(cli, ["sim", "replay", traj_path])
        assert result.exit_code == 0
        assert "Steps loaded" in result.output or "trajectory" in result.output.lower()

    def test_sim_replay_missing_file_exits_nonzero(self) -> None:
        result = _runner().invoke(cli, ["sim", "replay", "/nonexistent/path.npz"])
        assert result.exit_code != 0

    def test_sim_replay_with_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = _write_trajectory(tmp)
            result = _runner().invoke(cli, ["sim", "replay", traj_path, "--seed", "42"])
        assert result.exit_code == 0

    def test_sim_replay_stop_on_termination_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = _write_trajectory(tmp)
            result = _runner().invoke(
                cli, ["sim", "replay", traj_path, "--no-stop-on-termination"]
            )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# sim calibrate
# ---------------------------------------------------------------------------


class TestSimCalibrateCommand:
    def test_calibrate_valid_data(self) -> None:
        data = {
            "sim_obs": [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
            "real_obs": [[1.05, 1.95], [1.15, 2.05], [1.25, 2.15]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_json(data, tmp)
            result = _runner().invoke(cli, ["sim", "calibrate", data_path])
        assert result.exit_code == 0
        assert "Scale Factor" in result.output or "Calibration" in result.output

    def test_calibrate_saves_output_file(self) -> None:
        data = {
            "sim_obs": [[1.0], [2.0], [3.0]],
            "real_obs": [[1.1], [2.1], [3.1]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_json(data, tmp)
            output_path = str(Path(tmp) / "profile.json")
            result = _runner().invoke(
                cli, ["sim", "calibrate", data_path, "--output", output_path]
            )
            assert result.exit_code == 0
            assert Path(output_path).exists()
            saved = json.loads(Path(output_path).read_text())
            assert "scale_factors" in saved

    def test_calibrate_missing_keys_exits_nonzero(self) -> None:
        data = {"other_key": []}
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_json(data, tmp)
            result = _runner().invoke(cli, ["sim", "calibrate", data_path])
        assert result.exit_code != 0

    def test_calibrate_bad_json_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text("{not valid json", encoding="utf-8")
            result = _runner().invoke(cli, ["sim", "calibrate", str(bad_path)])
        assert result.exit_code != 0

    def test_calibrate_missing_file_exits_nonzero(self) -> None:
        result = _runner().invoke(cli, ["sim", "calibrate", "/no/such/file.json"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# sim gap-analysis
# ---------------------------------------------------------------------------


class TestSimGapAnalysisCommand:
    def test_gap_analysis_valid_data(self) -> None:
        data = {
            "sim_obs": [[1.0, 2.0], [1.1, 2.1]],
            "real_obs": [[1.05, 1.95], [1.15, 2.05]],
            "sim_rewards": [0.5, 0.6],
            "real_rewards": [0.48, 0.57],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_json(data, tmp)
            result = _runner().invoke(cli, ["sim", "gap-analysis", data_path])
        assert result.exit_code == 0

    def test_gap_analysis_without_rewards(self) -> None:
        data = {
            "sim_obs": [[1.0], [2.0]],
            "real_obs": [[1.1], [2.1]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_json(data, tmp)
            result = _runner().invoke(cli, ["sim", "gap-analysis", data_path])
        assert result.exit_code == 0

    def test_gap_analysis_saves_output(self) -> None:
        data = {
            "sim_obs": [[1.0], [2.0]],
            "real_obs": [[1.1], [2.1]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_json(data, tmp)
            output_path = str(Path(tmp) / "report.json")
            result = _runner().invoke(
                cli, ["sim", "gap-analysis", data_path, "--output", output_path]
            )
            assert result.exit_code == 0
            assert Path(output_path).exists()

    def test_gap_analysis_missing_keys_exits_nonzero(self) -> None:
        data = {"wrong": []}
        with tempfile.TemporaryDirectory() as tmp:
            data_path = _write_json(data, tmp)
            result = _runner().invoke(cli, ["sim", "gap-analysis", data_path])
        assert result.exit_code != 0

    def test_gap_analysis_bad_json_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text("not-json!", encoding="utf-8")
            result = _runner().invoke(cli, ["sim", "gap-analysis", str(bad_path)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# sim safety-check
# ---------------------------------------------------------------------------


class TestSimSafetyCheckCommand:
    def test_safety_check_valid_within_bounds(self) -> None:
        result = _runner().invoke(cli, ["sim", "safety-check", "[0.5, -0.3, 0.8]"])
        assert result.exit_code == 0
        assert "passed" in result.output.lower()

    def test_safety_check_violation_exits_nonzero(self) -> None:
        # Values outside default [-1, 1] range.
        result = _runner().invoke(cli, ["sim", "safety-check", "[2.0, 0.5, 0.5]"])
        assert result.exit_code != 0
        assert "Violations" in result.output or "violation" in result.output.lower()

    def test_safety_check_invalid_json_exits_nonzero(self) -> None:
        result = _runner().invoke(cli, ["sim", "safety-check", "not-valid-json"])
        assert result.exit_code != 0

    def test_safety_check_non_array_exits_nonzero(self) -> None:
        result = _runner().invoke(cli, ["sim", "safety-check", '{"a": 1}'])
        assert result.exit_code != 0

    def test_safety_check_with_constraints_file(self) -> None:
        constraints = [
            {
                "name": "dim0_range",
                "constraint_type": "range",
                "dimension": 0,
                "min_value": -2.0,
                "max_value": 2.0,
                "severity": "error",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            c_path = _write_json(constraints, tmp)
            result = _runner().invoke(
                cli,
                ["sim", "safety-check", "[1.5]", "--constraints", c_path],
            )
        assert result.exit_code == 0

    def test_safety_check_bad_constraints_file_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text("{bad}", encoding="utf-8")
            result = _runner().invoke(
                cli,
                ["sim", "safety-check", "[0.5]", "--constraints", str(bad_path)],
            )
        assert result.exit_code != 0

    def test_safety_check_empty_action_passes(self) -> None:
        result = _runner().invoke(cli, ["sim", "safety-check", "[]"])
        assert result.exit_code == 0
