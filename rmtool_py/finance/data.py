"""Synthetic factor-model returns with a KNOWN population covariance.

    returns = B · F + ε,   F ~ N(0, Σ_F) (K×T),   ε_i ~ N(0, idio_var_i)
    population  Σ = B Σ_F Bᵀ + diag(idio_var)

Returns ``(returns, Sigma_pop)`` so callers/tests have the exact population Σ.
With ``loadings=None`` the model is pure idiosyncratic, giving Σ = diag(idio_var)
— used by the gold-standard test to build a low-cardinality atomic spectrum.
"""

import numpy as np


def factor_model_returns(N, T, *, loadings=None, factor_cov=None, idio_var, seed=None):
    """Generate ``(returns N×T, Sigma_pop N×N)`` from a linear factor model."""
    rng = np.random.default_rng(seed)

    idio = np.asarray(idio_var, dtype=float)
    if idio.ndim == 0:
        idio = np.full(N, float(idio))
    if idio.shape != (N,):
        raise ValueError("idio_var must be a scalar or length-N array")

    eps = rng.standard_normal((N, T)) * np.sqrt(idio)[:, None]

    if loadings is None:
        returns = eps
        Sigma = np.diag(idio)
        return returns, Sigma

    B = np.asarray(loadings, dtype=float)
    if B.ndim != 2 or B.shape[0] != N:
        raise ValueError("loadings must be an N×K array")
    K = B.shape[1]

    Sigma_F = np.eye(K) if factor_cov is None else np.asarray(factor_cov, dtype=float)
    if Sigma_F.shape != (K, K):
        raise ValueError("factor_cov must be K×K matching loadings' K")

    L = np.linalg.cholesky(Sigma_F)
    F = L @ rng.standard_normal((K, T))
    returns = B @ F + eps
    Sigma = B @ Sigma_F @ B.T + np.diag(idio)
    return returns, Sigma
