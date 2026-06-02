import sympy as sp
from rmtool_py.core.polynomial import BivariatePolynomial

m, z = sp.symbols("m z")


def test_clear_denominators_keeps_numerator():
    # m - 0.5/(1-z) - 0.5/(2-z)  ->  numerator after clearing
    bp = BivariatePolynomial(m - sp.Rational(1, 2) / (1 - z) - sp.Rational(1, 2) / (2 - z), m, z)
    cleared = bp.clear_denominators()
    # No denominators left in m, z
    assert sp.denom(sp.together(cleared.expr)) == 1


def test_make_squarefree_removes_repeated_factor():
    bp = BivariatePolynomial((m - z) ** 2 * (m + 1), m, z)
    sf = bp.make_squarefree()
    # Square-free part wrt m: (m - z)(m + 1), proportional to that
    expected = (m - z) * (m + 1)
    assert sf.is_proportional_to(BivariatePolynomial(expected, m, z))


def test_normalize_is_idempotent():
    bp = BivariatePolynomial((m - z) ** 2 / (1 - z), m, z)
    once = bp.normalize()
    twice = once.normalize()
    assert once.is_proportional_to(twice)


def test_proportional_equality():
    a = BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert a.is_proportional_to(b)
    c = BivariatePolynomial(m ** 2 + m * z + 2, m, z)
    assert not a.is_proportional_to(c)
