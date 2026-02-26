"""TransferBridge — translate observations between simulation and reality.

The bridge applies a :class:`CalibrationProfile` to transform observations
and actions produced in one domain so they are appropriate for the other.
This is the core sim-to-real transfer primitive: it maps simulator units,
scales, and noise characteristics onto real-world distributions.

CalibrationProfile
------------------
Stores the per-dimension linear transform (scale + offset) discovered by
:class:`~agent_sim_bridge.transfer.calibration.Calibrator`, plus an
optional noise model name for post-transform noise injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationProfile:
    """Encodes a linear sim-to-real calibration transform.

    Each dimension ``i`` transforms as::

        real[i] = sim[i] * scale_factors[i] + offsets[i]

    The inverse (real-to-sim) is::

        sim[i] = (real[i] - offsets[i]) / scale_factors[i]

    Attributes
    ----------
    scale_factors:
        Per-dimension multiplicative scale.  Length must match the
        observation/action dimensionality the profile was fitted for.
    offsets:
        Per-dimension additive offset (applied after scaling).
    noise_model:
        Optional name of the noise model to apply after transformation.
        Must correspond to a key understood by
        :class:`~agent_sim_bridge.sensors.noise.NoiseModel`.  ``None``
        means no noise is added.
    dimension_names:
        Optional human-readable labels for each dimension.
    metadata:
        Arbitrary key/value annotations (e.g., calibration date, sensor
        serial numbers, environment version).
    """

    scale_factors: list[float]
    offsets: list[float]
    noise_model: str | None = None
    dimension_names: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.scale_factors) != len(self.offsets):
            raise ValueError(
                f"scale_factors length ({len(self.scale_factors)}) must equal "
                f"offsets length ({len(self.offsets)})."
            )
        if not self.scale_factors:
            raise ValueError("CalibrationProfile must have at least one dimension.")
        for i, scale in enumerate(self.scale_factors):
            if scale == 0.0:
                raise ValueError(
                    f"scale_factors[{i}] is zero, which would make the inverse "
                    "transform undefined."
                )

    @property
    def n_dims(self) -> int:
        """Number of dimensions this profile covers."""
        return len(self.scale_factors)


class TransferBridge:
    """Apply a :class:`CalibrationProfile` to translate between sim and real.

    The bridge is intentionally stateless beyond the profile — it does not
    accumulate history or track episodes.  Instantiate once per profile and
    reuse across steps.

    Parameters
    ----------
    profile:
        The calibration profile describing the linear transform.

    Raises
    ------
    ValueError
        If the profile has zero dimensions.

    Example
    -------
    ::

        profile = CalibrationProfile(scale_factors=[1.05, 0.98], offsets=[0.01, -0.02])
        bridge = TransferBridge(profile)
        real_obs = bridge.sim_to_real([1.0, 2.0])
        sim_obs = bridge.real_to_sim(real_obs)
    """

    def __init__(self, profile: CalibrationProfile) -> None:
        self._profile = profile

    @property
    def profile(self) -> CalibrationProfile:
        """The active :class:`CalibrationProfile`."""
        return self._profile

    def sim_to_real(self, sim_values: list[float]) -> list[float]:
        """Transform simulation values to their real-world equivalents.

        Applies ``real[i] = sim[i] * scale[i] + offset[i]`` element-wise.

        Parameters
        ----------
        sim_values:
            Per-dimension values in simulation units.  Length must equal
            :attr:`CalibrationProfile.n_dims`.

        Returns
        -------
        list[float]
            Values transformed to real-world units.

        Raises
        ------
        ValueError
            If ``sim_values`` length does not match the profile.
        """
        self._check_length(sim_values)
        return [
            value * scale + offset
            for value, scale, offset in zip(
                sim_values, self._profile.scale_factors, self._profile.offsets
            )
        ]

    def real_to_sim(self, real_values: list[float]) -> list[float]:
        """Transform real-world values back to simulation units.

        Applies ``sim[i] = (real[i] - offset[i]) / scale[i]`` element-wise.

        Parameters
        ----------
        real_values:
            Per-dimension values in real-world units.  Length must equal
            :attr:`CalibrationProfile.n_dims`.

        Returns
        -------
        list[float]
            Values in simulation units.

        Raises
        ------
        ValueError
            If ``real_values`` length does not match the profile.
        """
        self._check_length(real_values)
        return [
            (value - offset) / scale
            for value, scale, offset in zip(
                real_values, self._profile.scale_factors, self._profile.offsets
            )
        ]

    def _check_length(self, values: list[float]) -> None:
        if len(values) != self._profile.n_dims:
            raise ValueError(
                f"Expected {self._profile.n_dims} values, got {len(values)}."
            )

    def __repr__(self) -> str:
        return (
            f"TransferBridge(n_dims={self._profile.n_dims}, "
            f"noise_model={self._profile.noise_model!r})"
        )
