import sympy as sp
from rmtool_py.core.atoms import wigner_lmz
from rmtool_py.core.polynomial import BivariatePolynomial
from rmtool_py.core import encodings as enc

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
