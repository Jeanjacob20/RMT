import numpy as np

from rmtool_py import AlgebraicMeasure as AM
from rmtool_py.finance.correlation import correlation_matrix, remove_market_mode
from rmtool_py.finance.spectrum import mp_edges, information_eigenvalues
from rmtool_py.finance.eigenvectors import inverse_participation_ratio


def _predicted_cdf(measure, xs, gridmax):
    g = np.linspace(1e-4, gridmax, 2000)
    d = np.array(measure.density(g).density)
    cdf = np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(g))])
    cdf /= cdf[-1]
    return np.interp(xs, g, cdf)


def test_null_spectrum_matches_mp():
    c = 0.25
    N, T = 400, int(round(400 / c))
    rng = np.random.default_rng(0)
    C = correlation_matrix(rng.standard_normal((N, T))).C
    eigs = np.sort(np.linalg.eigvalsh(C))

    lo, hi = mp_edges(Q=T / N, sigma2=1.0)
    assert abs(eigs.min() - lo) < 0.05
    assert eigs.max() < hi + 0.15           # finite-N edge blur

    mp = AM.marchenko_pastur(c)
    Fpred = _predicted_cdf(mp, eigs, gridmax=3.0)
    ks = np.max(np.abs(np.arange(1, len(eigs) + 1) / len(eigs) - Fpred))
    assert ks < 0.05                        # measured ~0.006


def test_null_bulk_eigenvectors_delocalized():
    rng = np.random.default_rng(1)
    N, T = 300, 1500
    C = correlation_matrix(rng.standard_normal((N, T))).C
    _, V = np.linalg.eigh(C)
    iprs = np.array([inverse_participation_ratio(V[:, k]) for k in range(N)])
    assert abs(np.median(iprs) * N - 3.0) < 0.7


def test_market_mode_detected_and_removed():
    rng = np.random.default_rng(2)
    N, T = 200, 1000
    common = rng.standard_normal(T)
    R = 2.5 * np.outer(np.ones(N), common) + rng.standard_normal((N, T))
    C = correlation_matrix(R).C
    eigs = np.linalg.eigvalsh(C)

    _, hi = mp_edges(Q=T / N, sigma2=1.0)
    info = information_eigenvalues(eigs, hi)
    assert len(info) >= 1                    # market eigenvalue above the edge
    assert info.max() > 5 * hi               # >> lambda_+ (MP upper edge), as Laloux report

    mm = remove_market_mode(C)
    assert np.isclose(mm.eigval, eigs.max())          # the dominant eigenpair was removed
    # after removal the residual spectrum should sit inside the rescaled MP bulk
    _, hi_res = mp_edges(Q=T / N, sigma2=mm.sigma2_residual)
    assert np.linalg.eigvalsh(mm.deflated).max() < hi_res + 0.15
