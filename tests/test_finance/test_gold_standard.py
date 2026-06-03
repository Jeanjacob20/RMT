import numpy as np

from rmtool_py import AlgebraicMeasure as AM
from rmtool_py.finance.data import factor_model_returns
from rmtool_py.finance.correlation import correlation_matrix


def _predicted_cdf(measure, xs, gridmax):
    g = np.linspace(1e-4, gridmax, 2000)
    d = np.array(measure.density(g).density)
    cdf = np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(g))])
    cdf /= cdf[-1]
    return np.interp(xs, g, cdf)


def _ks(empirical_sorted, measure, gridmax):
    n = len(empirical_sorted)
    Fpred = _predicted_cdf(measure, empirical_sorted, gridmax)
    Femp = np.arange(1, n + 1) / n
    return float(np.max(np.abs(Femp - Fpred)))


def test_gold_standard_two_atom_population():
    # Population spectrum: half eigenvalues at 1.0, half at 3.0 (weights 1/2, 1/2).
    c = 0.3
    N, T = 400, int(round(400 / c))
    a, b = 1.0, 3.0
    idio = np.array([a] * (N // 2) + [b] * (N // 2))
    R, Sigma = factor_model_returns(N, T, loadings=None, idio_var=idio, seed=0)

    # population spectrum -> atomic measure -> deformed-MP sample prediction
    pop = AM.atomic([0.5, 0.5], [a, b])
    predicted = pop.times_wishart(c)

    # standardize=False (covariance): keeps the two-atom population intact;
    # standardize=True would normalize rows to unit variance and erase it.
    eigs = np.sort(np.linalg.eigvalsh(correlation_matrix(R, standardize=False).C))
    ks = _ks(eigs, predicted, gridmax=7.0)
    assert ks < 0.05                      # measured ~0.005


def test_gold_standard_plain_mp_special_case():
    # Sigma = I  ->  prediction collapses to plain Marčenko–Pastur.
    c = 0.25
    N, T = 400, int(round(400 / c))
    R, Sigma = factor_model_returns(N, T, loadings=None, idio_var=1.0, seed=1)
    assert np.allclose(Sigma, np.eye(N))

    predicted = AM.atomic([1.0], [1.0]).times_wishart(c)
    assert predicted == AM.marchenko_pastur(c)        # exact symbolic identity

    eigs = np.sort(np.linalg.eigvalsh(correlation_matrix(R, standardize=False).C))
    ks = _ks(eigs, predicted, gridmax=3.0)
    assert ks < 0.05
