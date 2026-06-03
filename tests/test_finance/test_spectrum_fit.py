import numpy as np

from rmtool_py.finance.correlation import correlation_matrix
from rmtool_py.finance.spectrum import (
    fit_marchenko_pastur, information_eigenvalues, mp_edges, empirical_density,
)


def _iid_corr_eigs(N, T, seed):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((N, T))
    C = correlation_matrix(R).C
    return np.linalg.eigvalsh(C)


def test_fit_recovers_unit_sigma2_on_null():
    N, T = 400, 1600              # Q=4, c=0.25
    eigs = _iid_corr_eigs(N, T, seed=0)
    fit = fit_marchenko_pastur(eigs, Q=T / N)
    assert abs(fit.sigma2_lsq - 1.0) < 0.1
    assert abs(fit.sigma2_market - 1.0) < 0.1
    assert fit.bulk_eig_fraction > 0.95


def test_fit_market_estimator_drops_with_spike():
    rng = np.random.default_rng(1)
    N, T = 200, 800
    common = rng.standard_normal(T)
    R = 2.5 * np.outer(np.ones(N), common) + rng.standard_normal((N, T))
    eigs = np.linalg.eigvalsh(correlation_matrix(R).C)
    fit = fit_marchenko_pastur(eigs, Q=T / N)
    assert fit.sigma2_market < 0.9          # market share removed
    assert fit.bulk_var_fraction < 1.0


def test_information_eigenvalues_above_edge():
    eigs = np.array([0.1, 0.5, 1.0, 1.8, 9.0, 25.0])
    _, hi = mp_edges(Q=4.0, sigma2=1.0)     # hi = 2.25
    info = information_eigenvalues(eigs, hi)
    assert set(np.round(info, 3)) == {9.0, 25.0}


def test_empirical_density_hist_integrates_to_one():
    eigs = _iid_corr_eigs(300, 1200, seed=2)
    centers, heights = empirical_density(eigs, method="hist", bins=50)
    width = centers[1] - centers[0]
    assert abs(np.sum(heights) * width - 1.0) < 0.05
