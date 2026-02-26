"""NoiseModel — inject synthetic noise into sensor readings.

Provides two concrete noise models:

* :class:`GaussianNoise` — additive white Gaussian noise with configurable
  per-dimension standard deviations.
* :class:`UniformNoise` — additive noise sampled uniformly from
  ``[-magnitude, magnitude]`` per dimension.

Both use the Python stdlib ``random`` module — no external dependencies.

Usage
-----
::

    noise = GaussianNoise(std_devs=[0.01, 0.02])
    noisy_values = noise.apply([1.0, 2.0])
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod


class NoiseModel(ABC):
    """Abstract base class for sensor noise injection.

    Subclasses must implement :meth:`apply`.
    """

    @abstractmethod
    def apply(self, values: list[float]) -> list[float]:
        """Return a noisy copy of ``values``.

        Parameters
        ----------
        values:
            Clean sensor reading values.

        Returns
        -------
        list[float]
            Values with noise added.
        """

    def __call__(self, values: list[float]) -> list[float]:
        """Allow instances to be called directly as ``noise(values)``."""
        return self.apply(values)


class GaussianNoise(NoiseModel):
    """Additive white Gaussian noise.

    For each dimension ``i`` the output is::

        noisy[i] = values[i] + N(mean, std_devs[i])

    Parameters
    ----------
    std_devs:
        Per-dimension standard deviations.  If the input has more
        dimensions than ``std_devs``, the last std_dev is reused for all
        extra dimensions.  If ``std_devs`` is empty or all zeros, this
        model is effectively a pass-through.
    mean:
        Bias (mean of the Gaussian).  Defaults to 0.0 (zero-mean noise).
    seed:
        Optional RNG seed for reproducibility.

    Raises
    ------
    ValueError
        If ``std_devs`` is empty.
    """

    def __init__(
        self,
        std_devs: list[float],
        mean: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not std_devs:
            raise ValueError("GaussianNoise requires at least one std_dev value.")
        self._std_devs = list(std_devs)
        self._mean = mean
        self._rng = random.Random(seed)

    def apply(self, values: list[float]) -> list[float]:
        """Apply Gaussian noise to each element of ``values``.

        Parameters
        ----------
        values:
            Clean sensor reading.

        Returns
        -------
        list[float]
            Noisy copy.
        """
        result: list[float] = []
        for i, value in enumerate(values):
            std = self._std_devs[min(i, len(self._std_devs) - 1)]
            noise = self._rng.gauss(self._mean, std) if std != 0.0 else 0.0
            result.append(value + noise)
        return result

    def __repr__(self) -> str:
        return f"GaussianNoise(std_devs={self._std_devs!r}, mean={self._mean})"


class UniformNoise(NoiseModel):
    """Additive uniform noise.

    For each dimension ``i`` the output is::

        noisy[i] = values[i] + U(-magnitudes[i], magnitudes[i])

    Parameters
    ----------
    magnitudes:
        Per-dimension half-width of the uniform distribution.  If the
        input has more dimensions than ``magnitudes``, the last value is
        reused.  A magnitude of 0.0 means no noise for that dimension.
    seed:
        Optional RNG seed for reproducibility.

    Raises
    ------
    ValueError
        If ``magnitudes`` is empty.
    """

    def __init__(
        self,
        magnitudes: list[float],
        seed: int | None = None,
    ) -> None:
        if not magnitudes:
            raise ValueError("UniformNoise requires at least one magnitude value.")
        self._magnitudes = list(magnitudes)
        self._rng = random.Random(seed)

    def apply(self, values: list[float]) -> list[float]:
        """Apply uniform noise to each element of ``values``.

        Parameters
        ----------
        values:
            Clean sensor reading.

        Returns
        -------
        list[float]
            Noisy copy.
        """
        result: list[float] = []
        for i, value in enumerate(values):
            magnitude = self._magnitudes[min(i, len(self._magnitudes) - 1)]
            noise = self._rng.uniform(-magnitude, magnitude) if magnitude != 0.0 else 0.0
            result.append(value + noise)
        return result

    def __repr__(self) -> str:
        return f"UniformNoise(magnitudes={self._magnitudes!r})"


class CompositeNoise(NoiseModel):
    """Apply multiple noise models sequentially.

    Parameters
    ----------
    models:
        Noise models to apply in order.  The output of each model is fed
        as input to the next.

    Example
    -------
    ::

        noise = CompositeNoise([GaussianNoise([0.01]), UniformNoise([0.005])])
        noisy = noise.apply([1.0, 2.0])
    """

    def __init__(self, models: list[NoiseModel]) -> None:
        if not models:
            raise ValueError("CompositeNoise requires at least one model.")
        self._models = list(models)

    def apply(self, values: list[float]) -> list[float]:
        result = list(values)
        for model in self._models:
            result = model.apply(result)
        return result

    def __repr__(self) -> str:
        return f"CompositeNoise(models={self._models!r})"
