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


def test_normalize_preserves_irreducible_polynomial():
    # An irreducible encoding polynomial (the Wigner L_mz) must survive
    # normalize() unchanged, up to a nonzero constant.
    bp = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert bp.normalize().is_proportional_to(bp)


def test_proportional_equality():
    a = BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert a.is_proportional_to(b)
    c = BivariatePolynomial(m ** 2 + m * z + 2, m, z)
    assert not a.is_proportional_to(c)


def test_proportional_zero_cases():
    zero = BivariatePolynomial(0, m, z)
    nonzero = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert zero.is_proportional_to(BivariatePolynomial(0, m, z))
    assert not zero.is_proportional_to(nonzero)
    assert not nonzero.is_proportional_to(zero)


def test_clear_denominators_cancels_shared_factor():
    # -(m**2 + m*z + 1) / (m*z + 1) : numerator and denominator share no factor
    # here, but build one that does: (m+1)*(m-z) / (m+1) -> should reduce to (m-z)
    bp = BivariatePolynomial((m + 1) * (m - z) / (m + 1), m, z)
    cleared = bp.clear_denominators()
    assert cleared.is_proportional_to(BivariatePolynomial(m - z, m, z))


def test_make_squarefree_handles_zero():
    bp = BivariatePolynomial(0, m, z)
    assert bp.make_squarefree().expr == 0
    assert bp.normalize().expr == 0


def test_normalize_preserves_var2_independent_polynomial():
    # A polynomial independent of z must NOT be collapsed to a constant.
    bp = BivariatePolynomial(m ** 2 + 1, m, z)
    assert bp.normalize().is_proportional_to(bp)


def test_make_squarefree_var_independent_is_noop():
    bp = BivariatePolynomial(m ** 2 + 1, m, z)
    assert bp.make_squarefree(z).is_proportional_to(bp)


def test_canonical_form_collapses_constant_multiple():
    a = BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert a.canonical_form() == b.canonical_form()


def test_canonical_form_collapses_negative_and_fraction():
    a = BivariatePolynomial(-sp.Rational(2, 3) * (m ** 2 + m * z + 1), m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert a.canonical_form() == b.canonical_form()


def test_canonical_form_distinguishes_measures():
    a = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    c = BivariatePolynomial(m ** 2 + m * z + 2, m, z)
    assert a.canonical_form() != c.canonical_form()


def test_canonical_form_is_hashable_and_consistent():
    a = BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert hash(a.canonical_form()) == hash(b.canonical_form())
