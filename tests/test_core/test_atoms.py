import pytest
import sympy as sp
from rmtool_py.core.atoms import wigner_lmz, atomic_lmz, marchenko_pastur
from rmtool_py.core.polynomial import BivariatePolynomial

m, z = sp.symbols("m z")
c = sp.symbols("c")


def test_wigner_lmz():
    # Table 2(c): L_mz = m^2 + m z + 1
    assert wigner_lmz().is_proportional_to(BivariatePolynomial(m ** 2 + m * z + 1, m, z))


def test_atomic_two_point_eq_3_7():
    # Eq. 3.7: atoms at 0 and 1, weights 1/2, 1/2  ->  L_mz = m(2z^2 - 2z) - (1 - 2z)
    bp = atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 1])
    expected = BivariatePolynomial(m * (2 * z ** 2 - 2 * z) - (1 - 2 * z), m, z)
    assert bp.is_proportional_to(expected)


def test_atomic_single_point():
    # Single atom at a: m(z) = 1/(a - z)  ->  L_mz = m(a - z) - 1
    a = 3
    bp = atomic_lmz([1], [a])
    expected = BivariatePolynomial(m * (a - z) - 1, m, z)
    assert bp.is_proportional_to(expected)


def test_atomic_lmz_mismatched_lengths():
    with pytest.raises(ValueError):
        atomic_lmz([1], [0, 1])


def test_marchenko_pastur_table_2b():
    # Table 2(b): L_mz = c z m^2 - (1 - c - z) m + 1
    bp = marchenko_pastur(c)
    expected = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    assert bp.is_proportional_to(expected)


def test_marchenko_pastur_edges():
    # Density-support edges are roots of the m-discriminant; must equal (1 +/- sqrt(c))^2.
    cval = sp.Rational(1, 4)
    bp = marchenko_pastur(cval)
    poly_m = sp.Poly(bp.expr, m)
    a, b, cc = poly_m.all_coeffs()            # a m^2 + b m + cc
    disc = sp.expand(b ** 2 - 4 * a * cc)     # quadratic in z
    edges = sorted(sp.solve(disc, z))
    expected = sorted([(1 - sp.sqrt(cval)) ** 2, (1 + sp.sqrt(cval)) ** 2])
    assert [sp.nsimplify(e) for e in edges] == [sp.nsimplify(e) for e in expected]
