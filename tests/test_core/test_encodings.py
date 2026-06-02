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
