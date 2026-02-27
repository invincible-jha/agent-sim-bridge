"""Pure statistical functions for distribution comparison.

All implementations use only Python stdlib (math module).  No numpy or scipy
dependency.  Suitable for embedded use, CI environments, and production code
where dependency footprint must be minimised.

Functions
---------
kl_divergence
    KL(P || Q) = Σ p(i) * log(p(i) / q(i)), with zero-handling via additive
    smoothing (epsilon = 1e-10).
wasserstein_distance_1d
    Earth Mover's Distance for 1-D distributions: Σ |CDF_P - CDF_Q| * bin_width.
maximum_mean_discrepancy
    MMD with RBF kernel: k(a, b) = exp(-||a-b||² / (2 * bandwidth²)).
jensen_shannon_divergence
    JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q).
normalize_distribution
    Normalise a list of non-negative floats to sum to 1.0.
descriptive_stats
    Compute mean, std, min, max, median, q25, q75 for a sample.
"""
from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _smooth(values: list[float], epsilon: float = _EPSILON) -> list[float]:
    """Add *epsilon* to every element to avoid log(0) in KL computation.

    Parameters
    ----------
    values:
        Raw probability mass values (need not sum to 1).
    epsilon:
        Additive smoothing constant.

    Returns
    -------
    list[float]
        Smoothed values (not renormalised).
    """
    return [v + epsilon for v in values]


def _renormalize(values: list[float]) -> list[float]:
    """Divide each element by the total sum.

    Parameters
    ----------
    values:
        Non-negative floats.

    Returns
    -------
    list[float]
        Values that sum to 1.0.

    Raises
    ------
    ValueError
        If the sum is zero (all values are zero).
    """
    total = sum(values)
    if total == 0.0:
        raise ValueError(
            "Cannot normalise a distribution whose elements all sum to zero."
        )
    return [v / total for v in values]


def _cdf(distribution: list[float]) -> list[float]:
    """Compute the empirical CDF from a probability mass vector.

    Parameters
    ----------
    distribution:
        Probability mass vector (should sum to 1.0; treated as relative weights
        if it does not).

    Returns
    -------
    list[float]
        Cumulative distribution of the same length.
    """
    cumulative: list[float] = []
    running_total = 0.0
    for mass in distribution:
        running_total += mass
        cumulative.append(running_total)
    return cumulative


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Interpolated percentile for a pre-sorted list.

    Uses the linear interpolation method (equivalent to numpy's method=7).

    Parameters
    ----------
    sorted_values:
        Pre-sorted list of floats (ascending).
    fraction:
        Fraction in [0, 1] (e.g. 0.25 for Q1, 0.75 for Q3).

    Returns
    -------
    float
        The interpolated percentile value.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    index = fraction * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_values[n - 1]
    weight = index - lower
    return sorted_values[lower] + weight * (sorted_values[upper] - sorted_values[lower])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_distribution(values: list[float]) -> list[float]:
    """Normalise a list of non-negative floats so they sum to 1.0.

    Parameters
    ----------
    values:
        Non-negative floats representing counts, densities, or raw weights.

    Returns
    -------
    list[float]
        Values normalised to sum to 1.0.

    Raises
    ------
    ValueError
        If *values* is empty or all elements are zero.

    Examples
    --------
    >>> normalize_distribution([1.0, 2.0, 1.0])
    [0.25, 0.5, 0.25]
    """
    if not values:
        raise ValueError("Cannot normalise an empty distribution.")
    return _renormalize(values)


def kl_divergence(p: list[float], q: list[float]) -> float:
    """Kullback-Leibler divergence KL(P || Q).

    KL(P || Q) = Σ p(i) * log(p(i) / q(i))

    Zero-handling is performed by adding *epsilon* = 1e-10 to every element
    before normalising, preventing log(0) and division-by-zero.

    Parameters
    ----------
    p:
        Reference distribution (will be normalised internally).
    q:
        Approximate distribution (will be normalised internally).

    Returns
    -------
    float
        KL divergence ≥ 0.  Returns 0.0 for identical distributions.

    Raises
    ------
    ValueError
        If *p* and *q* have different lengths or are empty.

    Notes
    -----
    The result is not symmetric: KL(P || Q) ≠ KL(Q || P) in general.

    Examples
    --------
    >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
    0.0
    """
    if not p or not q:
        raise ValueError("Distributions must be non-empty.")
    if len(p) != len(q):
        raise ValueError(
            f"Distributions must have the same length; got {len(p)} and {len(q)}."
        )

    p_smooth = _renormalize(_smooth(p))
    q_smooth = _renormalize(_smooth(q))

    divergence = sum(
        pi * math.log(pi / qi) for pi, qi in zip(p_smooth, q_smooth)
    )
    return max(0.0, divergence)


def wasserstein_distance_1d(p: list[float], q: list[float]) -> float:
    """Wasserstein-1 (Earth Mover's) distance for 1-D distributions.

    Computed as the L1 distance between the cumulative distribution functions:

        W₁(P, Q) = Σ |CDF_P(i) - CDF_Q(i)| * bin_width

    where bin_width = 1 / n (uniform bins).

    Parameters
    ----------
    p:
        Probability mass vector for distribution P (will be normalised).
    q:
        Probability mass vector for distribution Q (will be normalised).

    Returns
    -------
    float
        Wasserstein-1 distance ≥ 0.  Returns 0.0 for identical distributions.

    Raises
    ------
    ValueError
        If *p* and *q* have different lengths or are empty.

    Examples
    --------
    >>> wasserstein_distance_1d([1.0, 0.0], [0.0, 1.0])
    0.5
    """
    if not p or not q:
        raise ValueError("Distributions must be non-empty.")
    if len(p) != len(q):
        raise ValueError(
            f"Distributions must have the same length; got {len(p)} and {len(q)}."
        )

    p_norm = _renormalize(_smooth(p))
    q_norm = _renormalize(_smooth(q))

    cdf_p = _cdf(p_norm)
    cdf_q = _cdf(q_norm)

    n = len(p_norm)
    bin_width = 1.0 / n

    distance = sum(abs(cp - cq) for cp, cq in zip(cdf_p, cdf_q)) * bin_width
    return max(0.0, distance)


def maximum_mean_discrepancy(
    x: list[float],
    y: list[float],
    bandwidth: float = 1.0,
) -> float:
    """Maximum Mean Discrepancy with an RBF (Gaussian) kernel.

    MMD²(X, Y) = E[k(x, x')] - 2 E[k(x, y)] + E[k(y, y')]

    where k(a, b) = exp(-||a - b||² / (2 * bandwidth²)).

    MMD is returned as the non-negative square root of MMD².

    Parameters
    ----------
    x:
        Samples from the first distribution.
    y:
        Samples from the second distribution.
    bandwidth:
        RBF kernel bandwidth σ (standard deviation of the Gaussian).
        Larger values make the kernel smoother / less sensitive to
        fine-grained differences.

    Returns
    -------
    float
        MMD ≥ 0.  Returns approximately 0 when X and Y are drawn from the
        same distribution.

    Raises
    ------
    ValueError
        If either sample is empty or *bandwidth* ≤ 0.

    Examples
    --------
    >>> maximum_mean_discrepancy([0.0, 1.0], [0.0, 1.0])  # same → ~0
    0.0
    """
    if not x or not y:
        raise ValueError("Both sample lists must be non-empty.")
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be > 0; got {bandwidth}.")

    two_bw_sq = 2.0 * bandwidth * bandwidth

    def rbf(a: float, b: float) -> float:
        diff = a - b
        return math.exp(-(diff * diff) / two_bw_sq)

    # E[k(x, x')]
    n = len(x)
    xx_sum = sum(rbf(x[i], x[j]) for i in range(n) for j in range(n))
    e_xx = xx_sum / (n * n)

    # E[k(y, y')]
    m = len(y)
    yy_sum = sum(rbf(y[i], y[j]) for i in range(m) for j in range(m))
    e_yy = yy_sum / (m * m)

    # E[k(x, y)]
    xy_sum = sum(rbf(xi, yj) for xi in x for yj in y)
    e_xy = xy_sum / (n * m)

    mmd_sq = e_xx - 2.0 * e_xy + e_yy
    # Numerical noise can yield tiny negatives; clamp to 0.
    return math.sqrt(max(0.0, mmd_sq))


def jensen_shannon_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence between two distributions.

    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    where M = 0.5 * (P + Q).

    JSD is symmetric and bounded in [0, log(2)] (using natural log).  It
    equals 0 iff P = Q.

    Parameters
    ----------
    p:
        First distribution (will be normalised internally).
    q:
        Second distribution (will be normalised internally).

    Returns
    -------
    float
        JSD ∈ [0, ln(2)] ≈ [0, 0.693].  Returns 0.0 for identical
        distributions.

    Raises
    ------
    ValueError
        If *p* and *q* have different lengths or are empty.

    Examples
    --------
    >>> jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5])
    0.0
    """
    if not p or not q:
        raise ValueError("Distributions must be non-empty.")
    if len(p) != len(q):
        raise ValueError(
            f"Distributions must have the same length; got {len(p)} and {len(q)}."
        )

    p_norm = _renormalize(_smooth(p))
    q_norm = _renormalize(_smooth(q))

    # Mixture distribution M
    mixture = [(pi + qi) / 2.0 for pi, qi in zip(p_norm, q_norm)]

    def _kl_pre_normalised(dist_a: list[float], dist_b: list[float]) -> float:
        """KL divergence between two already-normalised distributions."""
        return sum(
            ai * math.log(ai / bi) for ai, bi in zip(dist_a, dist_b) if ai > 0.0
        )

    jsd = 0.5 * _kl_pre_normalised(p_norm, mixture) + 0.5 * _kl_pre_normalised(
        q_norm, mixture
    )
    return max(0.0, jsd)


def descriptive_stats(values: list[float]) -> dict[str, float]:
    """Compute descriptive statistics for a sample.

    Parameters
    ----------
    values:
        Sample of floats.  Must be non-empty.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``mean``, ``std``, ``min``, ``max``,
        ``median``, ``q25``, ``q75``.

    Raises
    ------
    ValueError
        If *values* is empty.

    Examples
    --------
    >>> stats = descriptive_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> stats["mean"]
    3.0
    """
    if not values:
        raise ValueError("descriptive_stats requires at least one value.")

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    sorted_values = sorted(values)
    minimum = sorted_values[0]
    maximum = sorted_values[-1]
    median = _percentile(sorted_values, 0.5)
    q25 = _percentile(sorted_values, 0.25)
    q75 = _percentile(sorted_values, 0.75)

    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "median": median,
        "q25": q25,
        "q75": q75,
    }
