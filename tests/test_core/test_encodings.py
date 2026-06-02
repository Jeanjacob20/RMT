import pytest
import sympy as sp
from rmtool_py.core.atoms import wigner_lmz
from rmtool_py.core.polynomial import BivariatePolynomial
from rmtool_py.core import encodings as enc
from rmtool_py.core.algebra import boxplus

m, z, g, r, s, y = sp.symbols("m z g r s y")


def _prop(bp, expr, v1, v2):
    return bp.is_proportional_to(BivariatePolynomial(expr, v1, v2))


def test_wigner_encodings_gz_rg_sy():
    # Table 2(c) values for the four Phase-1 encodings (muz/etaz are Plan 2).
    lmz = wigner_lmz()
    assert _prop(enc.to_encoding(lmz, "gz"), g ** 2 - g * z + 1, g, z)
    assert _prop(enc.to_encoding(lmz, "rg"), r - g, r, g)
    assert _prop(enc.to_encoding(lmz, "sy"), s ** 2 * y - 1, s, y)


def test_roundtrip_mz_gz_mz():
    lmz = wigner_lmz()
    back = enc.to_encoding(enc.to_encoding(lmz, "gz"), "mz")
    assert back.is_proportional_to(lmz)


def test_roundtrip_mz_rg_mz():
    lmz = wigner_lmz()
    back = enc.to_encoding(enc.to_encoding(lmz, "rg"), "mz")
    assert back.is_proportional_to(lmz)


def test_roundtrip_mz_sy_mz():
    lmz = wigner_lmz()
    back = enc.to_encoding(enc.to_encoding(lmz, "sy"), "mz")
    assert back.is_proportional_to(lmz)


def test_to_encoding_mz_is_noop():
    lmz = wigner_lmz()
    assert enc.to_encoding(lmz, "mz").is_proportional_to(lmz)


def test_to_encoding_unknown_target_raises():
    with pytest.raises(ValueError):
        enc.to_encoding(wigner_lmz(), "xy")


def test_to_encoding_unknown_source_raises():
    q, w = sp.symbols("q w")
    bad = BivariatePolynomial(q + w, q, w)
    with pytest.raises(ValueError):
        enc.to_encoding(bad, "mz")


def test_free_sum_of_two_semicircles_is_degree_two():
    # L_rg of Wigner is r - g (linear). boxplus over g of two copies, back to mz.
    lmz = wigner_lmz()
    lrg = enc.to_encoding(lmz, "rg")          # vars (r, g)
    summed_rg = boxplus(lrg, lrg, r)          # add R-transforms over g
    back = enc.to_encoding(
        BivariatePolynomial(summed_rg.expr, r, g), "mz"
    )
    # free sum of two unit semicircles is a variance-2 semicircle: 2m^2 + mz + 1
    assert back.is_proportional_to(BivariatePolynomial(2 * m ** 2 + m * z + 1, m, z))


from rmtool_py.core.atoms import atomic_lmz

mu, eta = sp.symbols("mu eta")


def test_wigner_muz_etaz():
    lmz = wigner_lmz()
    assert _prop(enc.to_encoding(lmz, "muz"), mu ** 2 * z ** 2 - mu + 1, mu, z)
    assert _prop(enc.to_encoding(lmz, "etaz"), z ** 2 * eta ** 2 - eta + 1, eta, z)


def test_atomic_muz_etaz():
    # Table 2(a): atoms at 0 and 1, weights 1/2.
    lmz = atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 1])
    assert _prop(enc.to_encoding(lmz, "muz"), (-2 + 2 * z) * mu + 2 - z, mu, z)
    assert _prop(enc.to_encoding(lmz, "etaz"), (2 * z + 2) * eta - 2 - z, eta, z)


def test_mp_muz_etaz():
    # Marchenko-Pastur Table 2(b). NB: the etaz cell uses the corrected (+z) form;
    # the paper's printed (-z) cell is a typo (fails eta-transform Eq. 2.10).
    c = sp.symbols("c")
    lmz = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    assert _prop(enc.to_encoding(lmz, "muz"),
                 c * z * mu ** 2 - (z * c + 1 - z) * mu + 1, mu, z)
    assert _prop(enc.to_encoding(lmz, "etaz"),
                 c * z * eta ** 2 + (1 + z - c * z) * eta - 1, eta, z)


def test_roundtrip_mz_muz_mz():
    c = sp.symbols("c")
    lmz = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    back = enc.to_encoding(enc.to_encoding(lmz, "muz"), "mz")
    assert back.is_proportional_to(lmz)


def test_roundtrip_mz_etaz_mz():
    c = sp.symbols("c")
    lmz = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    back = enc.to_encoding(enc.to_encoding(lmz, "etaz"), "mz")
    assert back.is_proportional_to(lmz)
