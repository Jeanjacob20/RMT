import numpy as np

from rmtool_py import AlgebraicMeasure as AM
from rmtool_py.finance.spectrum import mp_edges, mp_density


def test_mp_edges_closed_form_exact():
    # Q=4, sigma2=1 -> 1/Q=0.25, edges (1+0.25 -/+ 1) = (0.25, 2.25)
    lo, hi = mp_edges(4.0, 1.0)
    assert np.isclose(lo, 0.25)
    assert np.isclose(hi, 2.25)


def test_mp_edges_match_engine_support():
    Q, s2 = 10 / 3.0, 0.74
    lo, hi = mp_edges(Q, s2)
    mp = AM.marchenko_pastur(1.0 / Q).scale(s2)
    grid = np.linspace(1e-3, hi * 1.4, 1500)
    dens = np.array(mp.density(grid).density)
    support = grid[dens > 1e-4]
    assert abs(support.min() - lo) < 0.02 * hi
    assert abs(support.max() - hi) < 0.02 * hi


def test_mp_density_integrates_to_one():
    Q, s2 = 3.0, 1.0
    lo, hi = mp_edges(Q, s2)
    grid = np.linspace(lo, hi, 4000)
    mass = np.trapz(mp_density(grid, Q, s2), grid)
    assert abs(mass - 1.0) < 1e-2


def test_mp_density_matches_engine_shape():
    # closed-form finance density vs engine density: small KS over the bulk
    Q, s2 = 4.0, 1.0
    lo, hi = mp_edges(Q, s2)
    grid = np.linspace(lo + 1e-3, hi - 1e-3, 400)
    eng = np.array(AM.marchenko_pastur(1.0 / Q).scale(s2).density(grid).density)
    clo = mp_density(grid, Q, s2)
    # compare normalized cumulative curves (CDF) -> KS-like distance
    def cdf(d):
        c = np.concatenate([[0], np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(grid))])
        return c / c[-1]
    ks = np.max(np.abs(cdf(eng) - cdf(clo)))
    assert ks < 0.02
