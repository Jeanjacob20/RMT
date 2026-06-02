"""Operational laws on the L_mz bivariate polynomial (Rao & Edelman Tables 7-9).

Each function takes and returns a BivariatePolynomial in (m, z). Deterministic
laws are direct substitutions; stochastic laws (Task 5) route through the rg / sy
encodings and the boxplus / boxtimes operators.
"""

import sympy as sp

from .polynomial import BivariatePolynomial
from . import encodings as enc
from .algebra import boxplus, boxtimes

m, z = sp.symbols("m z")


def _mz(expr):
    """Normalize an expression into an L_mz BivariatePolynomial."""
    return BivariatePolynomial(expr, m, z).normalize()


def shift(L, alpha):
    """A + alpha*I  (Table 7, "Translate"):  z -> z - alpha."""
    alpha = sp.sympify(alpha)
    return _mz(L.expr.subs(z, z - alpha, simultaneous=True))


def scale(L, alpha):
    """alpha*A  (Table 7, "Scale"):  m -> alpha*m, z -> z/alpha."""
    alpha = sp.sympify(alpha)
    return _mz(L.expr.subs({m: alpha * m, z: z / alpha}, simultaneous=True))


def mobius(L, p, q, r, s):
    """(p*A + q*I) / (r*A + s*I)  (Table 7, "Mobius"). Sequential subs as printed:
    alpha = (q - s z)/(p - r z); beta = 1/(p - r z); z -> -alpha; m -> ((m/beta) - r)/(s - r*alpha)."""
    p, q, r, s = map(sp.sympify, (p, q, r, s))
    alpha = (q - s * z) / (p - r * z)
    beta = 1 / (p - r * z)
    e = L.expr.subs(z, -alpha, simultaneous=True)
    e = e.subs(m, ((m / beta) - r) / (s - r * alpha), simultaneous=True)
    return _mz(e)


def inverse(L):
    """A^{-1}  (Table 7, "Invert") = mobius(0, 1, 1, 0)."""
    return mobius(L, 0, 1, 1, 0)


def square(L):
    """A^2  (Table 8(a)): boxplus over m of the two square-root branches."""
    sqz = sp.sqrt(z)
    L1 = _mz(L.expr.subs({z: sqz, m: 2 * m * sqz}, simultaneous=True))
    L2 = _mz(L.expr.subs({z: -sqz, m: -2 * m * sqz}, simultaneous=True))
    return _mz(boxplus(L1, L2, m).expr)


def transpose(L, c):
    """If A = X X' then B = X' X, with c = Size(A)/Size(B)  (Table 7, project/augment
    with the augmenting atom at 0):  m -> (1 - 1/c)/(0 - z) + m/c."""
    c = sp.sympify(c)
    mb = (1 - 1 / c) * (1 / (0 - z)) + m / c
    return _mz(L.expr.subs(m, mb, simultaneous=True))
