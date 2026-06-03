import sympy as sp
import numpy as np

from rmtool_py import compat as rt

m, z, c = sp.symbols("m z c")


def test_wishartpol_and_wignerpol():
    assert rt.wishartpol(sp.Rational(1, 2)).is_proportional_to(rt.wishartpol(sp.Rational(1, 2)))
    assert sp.Poly(rt.wignerpol().expr, m).degree() == 2


def test_quickstart_wishart_moments():
    # Sec 1.3: b = wishartpol(0.5); Lmz2MomS(b, 10).  MP(1/2): M1..M3 = 1, 1.5, 2.75.
    moms = rt.Lmz2MomS(rt.wishartpol(sp.Rational(1, 2)), 10)
    assert len(moms) == 10
    assert [sp.nsimplify(x) for x in moms[:3]] == [1, sp.Rational(3, 2), sp.Rational(11, 4)]


def test_quickstart_wigner_moments():
    moms = rt.Lmz2MomS(rt.wignerpol(), 10)
    assert [sp.nsimplify(x) for x in moms[:5]] == [0, 1, 0, 2, 0]


def test_quickstart_density_runs():
    info = rt.Lmz2pdf(rt.wishartpol(sp.Rational(1, 2)), list(np.arange(-0.05, 5.0, 0.05)))
    assert hasattr(info, "range") and hasattr(info, "density")
    assert len(info.range) == len(info.density)
    assert all(d >= 0.0 and np.isfinite(d) for d in info.density)


def test_quickstart_aplusb_symbolic_c():
    # Sec 1.3: b = AplusB(wignerpol, wishartpol(c)); degree 3 in m.
    b = rt.AplusB(rt.wignerpol(), rt.wishartpol(c))
    assert sp.Poly(b.expr, m).degree() == 3


def test_section_2_5_atimesb_runs():
    # Sec 2.5: b3 = AtimesB(wishartpol(c), wignerpol); then pretty/latex/TLmz.
    b3 = rt.AtimesB(rt.wishartpol(c), rt.wignerpol())
    assert isinstance(rt.latex(b3), str)
    assert isinstance(rt.pretty(b3), str)
    assert rt.TLmz(b3).shape == (5, 3)


def test_deterministic_compat_names():
    L = rt.wignerpol()
    # compat returns BivariatePolynomial (no __eq__); compare with is_proportional_to.
    assert rt.shiftA(rt.wignerpol(), 1).is_proportional_to(rt.shiftA(rt.wignerpol(), 1))
    for fn in (rt.invA, rt.squareA):
        assert isinstance(fn(L), type(L))
    assert isinstance(rt.scaleA(L, 2), type(L))
    assert isinstance(rt.mobiusA(L, 1, 0, 0, 1), type(L))
    assert isinstance(rt.transposeA(L, sp.Rational(1, 2)), type(L))
    assert isinstance(rt.AtimesWish(L, sp.Rational(1, 2)), type(L))
    assert isinstance(rt.AgramWish(L, c, sp.Symbol("s")), type(L))
    assert isinstance(rt.compressA(rt.wishartpol(sp.Rational(1, 4)), sp.Rational(1, 2)), type(L))
