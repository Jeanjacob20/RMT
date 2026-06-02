import sympy as sp
from rmtool_py.core.polynomial import BivariatePolynomial
from rmtool_py.core.algebra import companion_matrix, boxplus, boxtimes

u, v = sp.symbols("u v")

# Table 6 inputs:
#   L1(u,v) = u^2 + u(1 - v) + v^2
#   L2(u,v) = u^2 (v^2 - 3v + 1) + u(1 + v) + v^2
L1 = BivariatePolynomial(u ** 2 + u * (1 - v) + v ** 2, u, v)
L2 = BivariatePolynomial(u ** 2 * (v ** 2 - 3 * v + 1) + u * (1 + v) + v ** 2, u, v)


def _key(c):
    c = complex(c)
    return (round(c.real, 6), round(c.imag, 6))


def _roots_at(bp, v0):
    p = sp.Poly(bp.expr.subs(v, v0), u)
    return sorted((complex(c) for c in p.all_roots()), key=_key)


def _sorted_complex(vals):
    return sorted((complex(c) for c in vals), key=_key)


def test_companion_charpoly_recovers_polynomial():
    C = companion_matrix(L1, u)
    # char poly det(u I - C) is proportional to L1 (monic in u)
    n = C.shape[0]
    chi = (u * sp.eye(n) - C).det()
    assert BivariatePolynomial(chi, u, v).is_proportional_to(L1)


def test_boxplus_values_are_sums_of_roots():
    L3 = boxplus(L1, L2, u)
    v0 = 2  # generic: L2's leading coeff v**2 - 3*v + 1 is nonzero here
    r1 = _roots_at(L1, v0)
    r2 = _roots_at(L2, v0)
    expected = _sorted_complex([a + b for a in r1 for b in r2])
    got = _roots_at(L3, v0)
    assert len(got) == len(expected)
    assert len(got) == 4  # deg(L1) * deg(L2) = 2 * 2
    for a, b in zip(got, expected):
        assert abs(a - b) < 1e-6


def test_boxtimes_values_are_products_of_roots():
    L3 = boxtimes(L1, L2, u)
    v0 = 2  # generic: L2's leading coeff v**2 - 3*v + 1 is nonzero here
    r1 = _roots_at(L1, v0)
    r2 = _roots_at(L2, v0)
    expected = _sorted_complex([a * b for a in r1 for b in r2])
    got = _roots_at(L3, v0)
    assert len(got) == len(expected)
    assert len(got) == 4  # deg(L1) * deg(L2) = 2 * 2
    for a, b in zip(got, expected):
        assert abs(a - b) < 1e-6


def test_boxplus_commutative():
    v0 = 2
    a = _roots_at(boxplus(L1, L2, u), v0)
    b = _roots_at(boxplus(L2, L1, u), v0)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-6
