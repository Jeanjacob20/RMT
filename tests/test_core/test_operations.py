import sympy as sp

from rmtool_py.core.polynomial import BivariatePolynomial
from rmtool_py.core.atoms import wigner_lmz, atomic_lmz, marchenko_pastur
from rmtool_py.core import operations as ops

m, z, c = sp.symbols("m z c")


def _atom(a):
    return atomic_lmz([1], [a])


def _prop(bp, other):
    return bp.is_proportional_to(other)


def test_shift_atom():
    assert _prop(ops.shift(_atom(3), 2), _atom(5))


def test_scale_atom():
    assert _prop(ops.scale(_atom(3), 2), _atom(6))


def test_inverse_atom():
    assert _prop(ops.inverse(_atom(3)), _atom(sp.Rational(1, 3)))


def test_square_atom():
    assert _prop(ops.square(_atom(3)), _atom(9))


def test_mobius_reduces_to_shift():
    # mobius(1, alpha, 0, 1) == shift(alpha): (1*A + alpha I)/(0*A + I) = A + alpha I
    assert _prop(ops.mobius(_atom(3), 1, 2, 0, 1), ops.shift(_atom(3), 2))


def test_transpose_atom_adds_zero_mass():
    # transpose(delta_3, c=1/2) == 1/2 delta_0 + 1/2 delta_3
    out = ops.transpose(_atom(3), sp.Rational(1, 2))
    expected = atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 3])
    assert _prop(out, expected)


def test_add_free_sum_eq_1_2():
    # Eq. 1.2: A = Wigner, B = MP(1/2). L_mz^{A+B} = m^3 + (z+2)m^2 + (2z-1)m + 2.
    out = ops.add(wigner_lmz(), marchenko_pastur(sp.Rational(1, 2)))
    expected = BivariatePolynomial(
        m ** 3 + (z + 2) * m ** 2 + (2 * z - 1) * m + 2, m, z)
    assert _prop(out, expected)


def test_mult_free_product_eq_1_4():
    # Eq. 1.4: L_mz^{AB} = m^4 z^2 - 2 m^3 z + m^2 + 4 m z + 4  (up to a constant).
    out = ops.mult(wigner_lmz(), marchenko_pastur(sp.Rational(1, 2)))
    expected = BivariatePolynomial(
        m ** 4 * z ** 2 - 2 * m ** 3 * z + m ** 2 + 4 * m * z + 4, m, z)
    assert _prop(out, expected)


def test_times_wishart_matches_eq_1_4():
    # Direct formula (Table 7) must match Eq. 1.4 = mult(Wigner, MP(1/2)),
    # confirming times_wishart is the specialised form of free multiplication.
    out = ops.times_wishart(wigner_lmz(), sp.Rational(1, 2))
    expected = BivariatePolynomial(
        m ** 4 * z ** 2 - 2 * m ** 3 * z + m ** 2 + 4 * m * z + 4, m, z)
    assert _prop(out, expected)


def test_times_wishart_equals_mult_with_mp():
    # A x W(c) is the free multiplicative convolution of A with MP(c).
    lhs = ops.times_wishart(wigner_lmz(), sp.Rational(1, 2))
    rhs = ops.mult(wigner_lmz(), marchenko_pastur(sp.Rational(1, 2)))
    assert _prop(lhs, rhs)


def test_gram_wishart_of_zero_atom_is_scaled_mp():
    # (sqrt(A) + sqrt(s) G)(...)' with A = delta_0 is s * W(c) = MP(c).scale(s).
    s = sp.symbols("s")
    out = ops.gram_wishart(atomic_lmz([1], [0]), c, s)
    expected = ops.scale(marchenko_pastur(c), s)
    assert _prop(out, expected)


def test_compress_atomic_eq_10_9():
    # Sec 10.2 worked example: compress(1/2 delta_0 + 1/2 delta_1, c).
    out = ops.compress(atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 1]), c)
    expected = BivariatePolynomial(
        (-2 * c * z ** 2 + 2 * c * z) * m ** 2
        - (-2 * c + 4 * c * z + 1 - 2 * z) * m
        + (-2 * c + 2), m, z)
    assert _prop(out, expected)
