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
