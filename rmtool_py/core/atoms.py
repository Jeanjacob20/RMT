"""Constructors for atomic and Wigner measure encodings (L_mz form)."""

import sympy as sp

from .polynomial import BivariatePolynomial

m, z = sp.symbols("m z")


def wigner_lmz():
    """Semicircle (Wigner) law. Table 2(c): L_mz = m^2 + m z + 1."""
    return BivariatePolynomial(m ** 2 + m * z + 1, m, z)


def atomic_lmz(weights, points):
    """Atomic measure sum_i w_i * delta(x - p_i).

    Stieltjes transform m(z) = sum_i w_i / (p_i - z); the L_mz polynomial is the
    numerator of (m - m(z)) after clearing denominators.
    """
    if len(weights) != len(points):
        raise ValueError("weights and points must have equal length")
    stieltjes = sum((w / (p - z) for w, p in zip(weights, points)), sp.Integer(0))
    bp = BivariatePolynomial(m - stieltjes, m, z)
    return bp.normalize()


def marchenko_pastur(c):
    """Marcenko-Pastur law with aspect ratio ``c`` (Table 2(b)).

    L_mz = c z m^2 - (1 - c - z) m + 1.  The density support is
    [(1 - sqrt(c))^2, (1 + sqrt(c))^2]; ``c`` may be numeric or symbolic.
    For Laloux's normalization use c = N/T = 1/Q and scale by sigma^2.
    """
    c = sp.sympify(c)
    return BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
