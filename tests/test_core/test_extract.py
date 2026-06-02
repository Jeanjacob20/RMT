import pytest
import sympy as sp

from rmtool_py.core.atoms import wigner_lmz, marchenko_pastur
from rmtool_py.core import extract

c = sp.symbols("c")


def test_moments_series_wigner_catalan():
    # Semicircle: M_1..M_7 = 0,1,0,2,0,5,0.
    got = extract.moments(wigner_lmz(), 7, method="series")
    assert [sp.nsimplify(x) for x in got] == [0, 1, 0, 2, 0, 5, 0]


def test_moments_series_mp_narayana():
    # MP(c): M_1..M_4 = 1, 1+c, 1+3c+c^2, 1+6c+6c^2+c^3.
    got = extract.moments(marchenko_pastur(c), 4, method="series")
    expected = [sp.Integer(1), 1 + c, 1 + 3 * c + c ** 2,
                1 + 6 * c + 6 * c ** 2 + c ** 3]
    assert [sp.expand(x) for x in got] == [sp.expand(e) for e in expected]


def test_moments_fast_matches_series_wigner():
    fast = extract.moments(wigner_lmz(), 7, method="fast")
    slow = extract.moments(wigner_lmz(), 7, method="series")
    assert [sp.nsimplify(a) for a in fast] == [sp.nsimplify(b) for b in slow]


def test_moments_fast_matches_series_mp_numeric():
    mp = marchenko_pastur(sp.Rational(1, 2))
    fast = extract.moments(mp, 6, method="fast")
    slow = extract.moments(mp, 6, method="series")
    assert [sp.nsimplify(a) for a in fast] == [sp.nsimplify(b) for b in slow]


def test_moments_fast_falls_back_to_series(monkeypatch):
    import sympy.holonomic as sh
    def _boom(*a, **kw):
        raise NotImplementedError("forced failure")
    monkeypatch.setattr(sh, "expr_to_holonomic", _boom)
    with pytest.warns(UserWarning):
        result = extract.moments(wigner_lmz(), 4, method="fast")
    assert [sp.nsimplify(x) for x in result] == [0, 1, 0, 2]


def test_moments_invalid_method():
    with pytest.raises(ValueError, match="method"):
        extract.moments(wigner_lmz(), 3, method="bogus")


import numpy as np


def test_density_wigner_peak_at_zero():
    # Semicircle rho(0) = sqrt(4)/ (2 pi) = 1/pi.
    info = extract.density(wigner_lmz(), [0.0])
    assert abs(info.density[0] - 1.0 / np.pi) < 1e-3


def test_density_wigner_zero_outside_support():
    info = extract.density(wigner_lmz(), [3.0])     # support is [-2, 2]
    assert abs(info.density[0]) < 1e-6


def test_density_mp_support_within_edges():
    # MP(1/4): support [(1-1/2)^2, (1+1/2)^2] = [0.25, 2.25].
    grid = list(np.arange(-0.5, 3.0, 0.02))
    info = extract.density(marchenko_pastur(sp.Rational(1, 4)), grid)
    positive = [x for x, d in zip(info.range, info.density) if d > 1e-4]
    assert min(positive) > 0.24 - 0.05
    assert max(positive) < 2.25 + 0.05


def test_density_returns_all_roots():
    info = extract.density(wigner_lmz(), [0.0])
    assert info.all_roots is not None and len(info.all_roots[0]) == 2
