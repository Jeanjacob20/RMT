"""Empirical spectrum, Marčenko–Pastur fit, and information eigenvalues (Laloux).

Closed-form MP (their Eq. 3), with c = N/T = 1/Q:
    edges  λ± = σ²(1 + 1/Q ± 2√(1/Q))
    ρ(λ)   = (Q / 2πσ²) · √((λ₊ − λ)(λ − λ₋)) / λ   on [λ₋, λ₊], else 0.
These are pinned by tests to agree with core.marchenko_pastur(1/Q).scale(σ²).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class MPFit:
    sigma2_market: float        # Laloux 1 - lambda_max/N
    sigma2_lsq: float           # least-squares best fit over the bulk
    edges: tuple                # (lambda_-, lambda_+) at sigma2_lsq
    bulk_eig_fraction: float    # fraction of eigenvalues inside [lambda_-, lambda_+]
    bulk_var_fraction: float    # fraction of total variance inside the bulk


def mp_edges(Q, sigma2=1.0):
    """Marčenko–Pastur support edges (λ₋, λ₊) for ratio c = 1/Q, scale σ²."""
    r = 1.0 / Q
    s = np.sqrt(r)
    return sigma2 * (1.0 + r - 2.0 * s), sigma2 * (1.0 + r + 2.0 * s)


def _mp_density(lam, Q, sigma2=1.0):
    """Closed-form MP density on the grid ``lam`` (0 outside the support)."""
    lam = np.asarray(lam, dtype=float)
    lo, hi = mp_edges(Q, sigma2)
    out = np.zeros_like(lam)
    inside = (lam > lo) & (lam < hi)
    l = lam[inside]
    out[inside] = (Q / (2.0 * np.pi * sigma2)) * np.sqrt((hi - l) * (l - lo)) / l
    return out


def empirical_density(eigs, *, method="hist", bins=50):
    """Empirical spectral density: returns (centers, heights).

    method="hist" -> normalized histogram; method="kde" -> Gaussian KDE sampled
    on a grid spanning the eigenvalues.
    """
    eigs = np.asarray(eigs, dtype=float)
    if method == "hist":
        heights, edges = np.histogram(eigs, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, heights
    if method == "kde":
        from scipy.stats import gaussian_kde
        centers = np.linspace(eigs.min(), eigs.max(), bins)
        return centers, gaussian_kde(eigs)(centers)
    raise ValueError("method must be 'hist' or 'kde'")


def fit_marchenko_pastur(eigs, Q):
    """Fit σ² to the eigenvalue bulk by both Laloux estimators -> :class:`MPFit`.

    Assumes ``eigs`` are eigenvalues of a correlation matrix (trace = N), so the
    market estimator σ² = 1 − λ_max/N is meaningful.
    """
    eigs = np.sort(np.asarray(eigs, dtype=float))
    N = len(eigs)
    sigma2_market = 1.0 - eigs[-1] / N

    def loss(s2):
        lo, hi = mp_edges(Q, s2)
        bulk = eigs[(eigs >= lo) & (eigs <= hi)]
        if len(bulk) < 10:
            return 1e6
        nbins = min(40, max(10, len(bulk) // 3))
        heights, edges = np.histogram(bulk, bins=nbins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return float(np.mean((heights - _mp_density(centers, Q, s2)) ** 2))

    res = minimize_scalar(loss, bounds=(0.05, 3.0), method="bounded")
    sigma2_lsq = float(res.x)
    lo, hi = mp_edges(Q, sigma2_lsq)
    mask = (eigs >= lo) & (eigs <= hi)
    return MPFit(
        sigma2_market=sigma2_market,
        sigma2_lsq=sigma2_lsq,
        edges=(lo, hi),
        bulk_eig_fraction=float(mask.mean()),
        bulk_var_fraction=float(eigs[mask].sum() / eigs.sum()),
    )


def information_eigenvalues(eigs, lambda_plus):
    """Eigenvalues above the upper MP edge ('signal')."""
    eigs = np.asarray(eigs, dtype=float)
    return eigs[eigs > lambda_plus]
