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
