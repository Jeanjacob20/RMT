import sympy as sp

from rmtool_py.core.measure import AlgebraicMeasure as AM

m, z, c = sp.symbols("m z c")


def test_constructors_and_eq_proportional():
    a = AM.wigner()
    # 2*(m^2 + m z + 1) is the same measure (proportional L_mz).
    from rmtool_py.core.polynomial import BivariatePolynomial
    b = AM(BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z))
    assert a == b
    assert hash(a) == hash(b)


def test_distinct_measures_not_equal():
    assert AM.wigner() != AM.marchenko_pastur(sp.Rational(1, 2))


def test_operator_add_is_free_sum_eq_1_2():
    out = AM.wigner() + AM.marchenko_pastur(sp.Rational(1, 2))
    from rmtool_py.core.polynomial import BivariatePolynomial
    expected = BivariatePolynomial(
        m ** 3 + (z + 2) * m ** 2 + (2 * z - 1) * m + 2, m, z)
    assert out.lmz.is_proportional_to(expected)


def test_operator_mul_is_free_product_eq_1_4():
    out = AM.wigner() * AM.marchenko_pastur(sp.Rational(1, 2))
    from rmtool_py.core.polynomial import BivariatePolynomial
    expected = BivariatePolynomial(
        m ** 4 * z ** 2 - 2 * m ** 3 * z + m ** 2 + 4 * m * z + 4, m, z)
    assert out.lmz.is_proportional_to(expected)


def test_invert_and_scale_shift():
    a = AM.atomic([1], [3])
    assert (~a) == AM.atomic([1], [sp.Rational(1, 3)])
    assert a.scale(2) == AM.atomic([1], [6])
    assert a.shift(2) == AM.atomic([1], [5])
    assert a.square() == AM.atomic([1], [9])


def test_times_wishart_method():
    out = AM.wigner().times_wishart(sp.Rational(1, 2))
    expected = AM.wigner() * AM.marchenko_pastur(sp.Rational(1, 2))
    assert out == expected


def test_moments_and_density_and_latex():
    mp = AM.marchenko_pastur(sp.Rational(1, 2))
    assert [sp.nsimplify(x) for x in mp.moments(3)] == [1, sp.Rational(3, 2), sp.Rational(11, 4)]
    info = mp.density([1.0])
    assert info.density[0] >= 0.0
    assert isinstance(mp.latex(), str) and "m" in mp.latex()
