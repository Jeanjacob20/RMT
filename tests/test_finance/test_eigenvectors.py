import numpy as np

from rmtool_py.finance.eigenvectors import (
    component_distribution, porter_thomas_pdf, inverse_participation_ratio,
)


def test_component_distribution_normalized_to_N():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(64)
    v = v / np.linalg.norm(v)               # unit eigenvector
    u = component_distribution(v)
    assert np.isclose(np.sum(u ** 2), len(v))


def test_porter_thomas_is_standard_normal_pdf():
    assert np.isclose(porter_thomas_pdf(0.0), 1.0 / np.sqrt(2 * np.pi))
    grid = np.linspace(-8, 8, 4000)
    assert abs(np.trapz(porter_thomas_pdf(grid), grid) - 1.0) < 1e-3


def test_ipr_localized_vs_delocalized():
    e1 = np.zeros(100); e1[0] = 1.0
    assert np.isclose(inverse_participation_ratio(e1), 1.0)   # fully localized
    flat = np.ones(100) / np.sqrt(100)
    assert np.isclose(inverse_participation_ratio(flat), 1.0 / 100)  # fully spread


def test_ipr_of_goe_eigenvector_near_3_over_N():
    rng = np.random.default_rng(3)
    N = 500
    A = rng.standard_normal((N, N)); A = (A + A.T) / 2
    _, V = np.linalg.eigh(A)
    iprs = np.array([inverse_participation_ratio(V[:, k]) for k in range(N)])
    assert abs(np.median(iprs) * N - 3.0) < 0.6      # delocalized ~ 3/N
