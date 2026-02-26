"""Calibrator — fit a CalibrationProfile from paired sim/real observations.

The calibrator uses simple per-dimension ordinary least squares (implemented
manually without numpy) to find the linear transform

    real[i] = sim[i] * scale[i] + offset[i]

that minimises the mean absolute error across a set of paired samples.

Evaluation
----------
:meth:`Calibrator.evaluate_calibration` returns the mean absolute error
(MAE) between real observations and the profile's predictions.  Lower is
better; zero indicates a perfect linear fit.
"""
from __future__ import annotations

import logging

from agent_sim_bridge.transfer.bridge import CalibrationProfile

logger = logging.getLogger(__name__)


def _ols_slope_intercept(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, float]:
    """Return (slope, intercept) from ordinary least squares.

    Solves::

        slope = (n * sum(x*y) - sum(x) * sum(y)) /
                (n * sum(x^2) - sum(x)^2)
        intercept = (sum(y) - slope * sum(x)) / n

    Falls back to slope=1, intercept=0 if the system is singular (e.g., all
    x values identical).

    Parameters
    ----------
    x_values:
        Predictor values (simulation observations for one dimension).
    y_values:
        Target values (real-world observations for the same dimension).

    Returns
    -------
    tuple[float, float]
        ``(slope, intercept)`` of the fitted line.
    """
    n = len(x_values)
    if n == 0:
        return 1.0, 0.0

    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(xi * yi for xi, yi in zip(x_values, y_values))
    sum_xx = sum(xi * xi for xi in x_values)

    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        # Degenerate case: all x values identical — slope is undefined.
        logger.warning(
            "OLS denominator is zero (all x values identical). "
            "Defaulting to slope=1, intercept=mean(y)-mean(x)."
        )
        mean_x = sum_x / n
        mean_y = sum_y / n
        return 1.0, mean_y - mean_x

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


class Calibrator:
    """Fit and evaluate a :class:`~agent_sim_bridge.transfer.bridge.CalibrationProfile`.

    Each call to :meth:`calibrate` accepts paired lists of sim and real
    observation vectors and returns a :class:`~agent_sim_bridge.transfer.bridge.CalibrationProfile`
    whose scale/offset parameters minimise the per-dimension MAE.

    The calibrator also keeps track of the most recent fit so that
    :meth:`evaluate_calibration` can be called without re-fitting.

    Parameters
    ----------
    noise_model:
        Optional noise model name to embed in the returned profile.

    Example
    -------
    ::

        calibrator = Calibrator()
        profile = calibrator.calibrate(sim_observations, real_observations)
        error = calibrator.evaluate_calibration(sim_observations, real_observations)
    """

    def __init__(self, noise_model: str | None = None) -> None:
        self._noise_model = noise_model
        self._last_profile: CalibrationProfile | None = None

    def calibrate(
        self,
        sim_obs: list[list[float]],
        real_obs: list[list[float]],
    ) -> CalibrationProfile:
        """Fit a :class:`~agent_sim_bridge.transfer.bridge.CalibrationProfile` from paired samples.

        Uses per-dimension ordinary least squares: for each dimension ``i``
        it fits ``real[i] = sim[i] * scale[i] + offset[i]``.

        Parameters
        ----------
        sim_obs:
            List of simulation observation vectors.  Each inner list must
            have the same length (number of dimensions).
        real_obs:
            List of real-world observation vectors, paired with ``sim_obs``.
            Must have the same outer and inner lengths as ``sim_obs``.

        Returns
        -------
        CalibrationProfile
            Fitted profile with ``scale_factors`` and ``offsets``.

        Raises
        ------
        ValueError
            If ``sim_obs`` and ``real_obs`` have different lengths, or if
            any observation vectors have mismatched dimensionality.
        """
        if len(sim_obs) != len(real_obs):
            raise ValueError(
                f"sim_obs has {len(sim_obs)} samples but real_obs has {len(real_obs)}."
            )
        if not sim_obs:
            raise ValueError("Cannot calibrate from zero samples.")

        n_dims = len(sim_obs[0])
        for index, (sim_vec, real_vec) in enumerate(zip(sim_obs, real_obs)):
            if len(sim_vec) != n_dims:
                raise ValueError(
                    f"sim_obs[{index}] has {len(sim_vec)} dimensions; expected {n_dims}."
                )
            if len(real_vec) != n_dims:
                raise ValueError(
                    f"real_obs[{index}] has {len(real_vec)} dimensions; expected {n_dims}."
                )

        scale_factors: list[float] = []
        offsets: list[float] = []

        for dim in range(n_dims):
            x_col = [sim_obs[sample][dim] for sample in range(len(sim_obs))]
            y_col = [real_obs[sample][dim] for sample in range(len(real_obs))]
            slope, intercept = _ols_slope_intercept(x_col, y_col)
            # OLS slope=1, intercept=0 means identity — ensure scale is nonzero.
            scale_factors.append(slope if slope != 0.0 else 1.0)
            offsets.append(intercept)

        profile = CalibrationProfile(
            scale_factors=scale_factors,
            offsets=offsets,
            noise_model=self._noise_model,
        )
        self._last_profile = profile
        logger.info(
            "Calibrated %d-dim profile from %d samples (noise_model=%r).",
            n_dims,
            len(sim_obs),
            self._noise_model,
        )
        return profile

    def evaluate_calibration(
        self,
        sim_obs: list[list[float]],
        real_obs: list[list[float]],
        profile: CalibrationProfile | None = None,
    ) -> float:
        """Compute mean absolute error between real observations and predicted values.

        Uses the provided ``profile`` (or the last fitted profile if
        ``profile`` is ``None``) to transform ``sim_obs`` and compares the
        result to ``real_obs`` element-wise.

        Parameters
        ----------
        sim_obs:
            Simulation observation vectors.
        real_obs:
            Corresponding real-world observation vectors.
        profile:
            Profile to evaluate.  Defaults to the most recently fitted
            profile from :meth:`calibrate`.

        Returns
        -------
        float
            Mean absolute error across all samples and dimensions.

        Raises
        ------
        RuntimeError
            If no profile is available (neither passed nor previously fitted).
        ValueError
            If ``sim_obs`` and ``real_obs`` lengths differ.
        """
        active_profile = profile or self._last_profile
        if active_profile is None:
            raise RuntimeError(
                "No calibration profile available. Call calibrate() first or pass a profile."
            )
        if len(sim_obs) != len(real_obs):
            raise ValueError(
                f"sim_obs has {len(sim_obs)} samples but real_obs has {len(real_obs)}."
            )
        if not sim_obs:
            return 0.0

        from agent_sim_bridge.transfer.bridge import TransferBridge

        bridge = TransferBridge(active_profile)
        total_error = 0.0
        count = 0

        for sim_vec, real_vec in zip(sim_obs, real_obs):
            predicted = bridge.sim_to_real(sim_vec)
            for predicted_val, actual_val in zip(predicted, real_vec):
                total_error += abs(predicted_val - actual_val)
                count += 1

        return total_error / count if count > 0 else 0.0

    @property
    def last_profile(self) -> CalibrationProfile | None:
        """The most recently fitted profile, or ``None`` if not yet fitted."""
        return self._last_profile

    def __repr__(self) -> str:
        fitted = self._last_profile is not None
        return f"Calibrator(noise_model={self._noise_model!r}, fitted={fitted})"
