"""DomainRandomizer — apply structured parameter randomization to environments.

Domain randomization is a sim-to-real transfer technique that trains agents
across a distribution of environment configurations so that the real world
appears as just another sample.  This module provides:

* :class:`RandomizationConfig` — specifies one parameter to randomize.
* :class:`DomainRandomizer` — samples parameter sets and applies them to
  environments via a callable interface.

No external numerical libraries are required — stdlib ``random`` is used for
all sampling.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DistributionType(str, Enum):
    """Supported sampling distributions for domain randomization."""

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    LOG_UNIFORM = "log_uniform"
    CONSTANT = "constant"


@dataclass
class RandomizationConfig:
    """Specification for randomizing one environment parameter.

    Attributes
    ----------
    parameter_name:
        Dot-path name of the parameter to randomize, e.g.
        ``"physics.gravity"`` or ``"sensor.noise_std"``.
    distribution:
        Sampling distribution to use.
    low:
        Lower bound for UNIFORM / LOG_UNIFORM, or mean - std for GAUSSIAN
        when combined with ``high``.  For CONSTANT this is the fixed value.
    high:
        Upper bound for UNIFORM / LOG_UNIFORM.  For GAUSSIAN this is the
        standard deviation; ``low`` then acts as the mean.
    clip_low:
        Optional hard minimum applied after sampling.
    clip_high:
        Optional hard maximum applied after sampling.
    metadata:
        Arbitrary annotations.
    """

    parameter_name: str
    distribution: DistributionType = DistributionType.UNIFORM
    low: float = 0.0
    high: float = 1.0
    clip_low: float | None = None
    clip_high: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def sample(self) -> float:
        """Draw one sample from this configuration's distribution.

        Returns
        -------
        float
            Sampled parameter value, clipped to ``[clip_low, clip_high]``
            if those bounds are set.
        """
        if self.distribution == DistributionType.CONSTANT:
            value = self.low
        elif self.distribution == DistributionType.UNIFORM:
            value = random.uniform(self.low, self.high)
        elif self.distribution == DistributionType.GAUSSIAN:
            # low = mean, high = std_dev
            value = random.gauss(self.low, self.high)
        elif self.distribution == DistributionType.LOG_UNIFORM:
            import math

            if self.low <= 0 or self.high <= 0:
                raise ValueError(
                    "LOG_UNIFORM requires low > 0 and high > 0, "
                    f"got low={self.low}, high={self.high}."
                )
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            value = math.exp(random.uniform(log_low, log_high))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution!r}")

        if self.clip_low is not None:
            value = max(self.clip_low, value)
        if self.clip_high is not None:
            value = min(self.clip_high, value)
        return value


class DomainRandomizer:
    """Sample and apply domain randomization configurations.

    Parameters
    ----------
    seed:
        Optional RNG seed for reproducible randomization.  When ``None``
        the system RNG state is used.

    Example
    -------
    ::

        configs = [
            RandomizationConfig("gravity", DistributionType.UNIFORM, low=8.0, high=12.0),
            RandomizationConfig("friction", DistributionType.GAUSSIAN, low=0.5, high=0.1),
        ]
        randomizer = DomainRandomizer(seed=42)
        params = randomizer.randomize(configs)
        # params == {"gravity": 9.43, "friction": 0.52}  (example values)
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    def randomize(self, configs: list[RandomizationConfig]) -> dict[str, float]:
        """Sample one value for each configuration.

        Parameters
        ----------
        configs:
            List of parameter specifications.  May be empty (returns ``{}``).

        Returns
        -------
        dict[str, float]
            Mapping of ``parameter_name`` to sampled value.
        """
        original_state = random.getstate()
        try:
            # Use the internal Random instance for reproducibility.
            result: dict[str, float] = {}
            for config in configs:
                # Temporarily swap stdlib RNG state to our seeded instance.
                random.setstate(self._rng.getstate())
                value = config.sample()
                self._rng.setstate(random.getstate())
                result[config.parameter_name] = value
                logger.debug(
                    "Randomized %r -> %.6f (%s)",
                    config.parameter_name,
                    value,
                    config.distribution.value,
                )
        finally:
            random.setstate(original_state)
        return result

    def apply_randomization(
        self,
        env: object,
        configs: list[RandomizationConfig],
    ) -> dict[str, float]:
        """Sample parameters and apply them to ``env`` via attribute assignment.

        The method traverses dot-path names in each config's
        ``parameter_name`` and sets the final attribute on the resolved
        object.  For example, a ``parameter_name`` of ``"physics.gravity"``
        calls ``setattr(env.physics, "gravity", value)``.

        Parameters
        ----------
        env:
            The environment object to modify in place.
        configs:
            Randomization configurations to apply.

        Returns
        -------
        dict[str, float]
            The parameter values that were applied (same as :meth:`randomize`).

        Raises
        ------
        AttributeError
            If a path segment does not exist on the target object.
        """
        sampled = self.randomize(configs)
        for parameter_name, value in sampled.items():
            parts = parameter_name.split(".")
            target = env
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
            logger.debug("Applied %r = %.6f to environment.", parameter_name, value)
        return sampled

    def __repr__(self) -> str:
        return f"DomainRandomizer(seed={self._seed!r})"
