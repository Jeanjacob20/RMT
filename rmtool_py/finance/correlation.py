"""Build correlation/covariance matrices from returns and remove the market mode.

Convention: ``C = (1/T) M Mᵀ`` from an N×T returns matrix (N assets, T obs).
``standardize=True`` (default) rescales each demeaned row to unit variance, so C
is a correlation matrix with unit diagonal; ``standardize=False`` returns the
(1/T-normalized) covariance.  Records N, T and Q = T/N.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Correlation:
    C: np.ndarray
    N: int
    T: int
    Q: float


@dataclass
class MarketMode:
    deflated: np.ndarray     # C with the market mode removed
    eigval: float            # market (largest) eigenvalue
    eigvec: np.ndarray       # market eigenvector
    sigma2_residual: float   # Laloux 1 - lambda_max / N


def correlation_matrix(returns, *, demean=True, standardize=True):
    """Return a :class:`Correlation` built as ``C = (1/T) M Mᵀ`` from N×T returns."""
    M = np.asarray(returns, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns must be a 2-D N×T array")
    N, T = M.shape
    if demean:
        M = M - M.mean(axis=1, keepdims=True)
    if standardize:
        std = np.sqrt((M ** 2).mean(axis=1, keepdims=True))   # population (1/T) std
        std[std == 0] = 1.0
        M = M / std
    C = (M @ M.T) / T
    return Correlation(C=C, N=N, T=T, Q=T / N)


def remove_market_mode(C):
    """Deflate the largest (market) eigenpair; return a :class:`MarketMode`."""
    C = np.asarray(C, dtype=float)
    N = C.shape[0]
    vals, vecs = np.linalg.eigh(C)            # ascending
    lam = vals[-1]
    v = vecs[:, -1]
    deflated = C - lam * np.outer(v, v)
    return MarketMode(deflated=deflated, eigval=lam, eigvec=v,
                      sigma2_residual=1.0 - lam / N)
