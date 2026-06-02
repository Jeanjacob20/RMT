"""AlgebraicMeasure: the object-oriented facade over the polynomial engine."""

import sympy as sp

from .polynomial import BivariatePolynomial
from . import atoms, operations, extract

m, z = sp.symbols("m z")


class AlgebraicMeasure:
    """A limiting eigenvalue distribution, carried as its canonical L_mz polynomial.

    Operators: a + b (free additive), a * b (free multiplicative), ~a (inverse).
    Equality is proportional (two proportional L_mz are the same measure) and hashable.
    """

    def __init__(self, lmz):
        if not isinstance(lmz, BivariatePolynomial):
            raise TypeError("AlgebraicMeasure expects a BivariatePolynomial in (m, z)")
        self.lmz = lmz.normalize()

    # --- constructors -----------------------------------------------------
    @classmethod
    def wigner(cls):
        return cls(atoms.wigner_lmz())

    @classmethod
    def marchenko_pastur(cls, c):
        return cls(atoms.marchenko_pastur(c))

    @classmethod
    def atomic(cls, weights, points):
        return cls(atoms.atomic_lmz(weights, points))

    # --- operators --------------------------------------------------------
    def __add__(self, other):
        return AlgebraicMeasure(operations.add(self.lmz, other.lmz))

    def __mul__(self, other):
        return AlgebraicMeasure(operations.mult(self.lmz, other.lmz))

    def __invert__(self):
        return AlgebraicMeasure(operations.inverse(self.lmz))

    # --- deterministic methods -------------------------------------------
    def scale(self, alpha):
        return AlgebraicMeasure(operations.scale(self.lmz, alpha))

    def shift(self, alpha):
        return AlgebraicMeasure(operations.shift(self.lmz, alpha))

    def mobius(self, p, q, r, s):
        return AlgebraicMeasure(operations.mobius(self.lmz, p, q, r, s))

    def square(self):
        return AlgebraicMeasure(operations.square(self.lmz))

    def transpose(self, c):
        return AlgebraicMeasure(operations.transpose(self.lmz, c))

    # --- stochastic methods ----------------------------------------------
    def times_wishart(self, c):
        return AlgebraicMeasure(operations.times_wishart(self.lmz, c))

    def gram_wishart(self, c, s):
        return AlgebraicMeasure(operations.gram_wishart(self.lmz, c, s))

    def compress(self, c):
        return AlgebraicMeasure(operations.compress(self.lmz, c))

    # --- extraction -------------------------------------------------------
    def density(self, grid, **kw):
        return extract.density(self.lmz, grid, **kw)

    def moments(self, k, method="series"):
        return extract.moments(self.lmz, k, method=method)

    # --- presentation -----------------------------------------------------
    def latex(self):
        return sp.latex(self.lmz.expr)

    def pretty(self):
        return sp.pretty(self.lmz.expr)

    def coefficient_table(self):
        """Matrix of coefficients of L_mz: entry [i, j] = coeff of m^i z^j (RMTool TLmz)."""
        poly = sp.Poly(self.lmz.expr, self.lmz.var1, self.lmz.var2)
        di = poly.degree(self.lmz.var1)
        dj = poly.degree(self.lmz.var2)
        return sp.Matrix(di + 1, dj + 1,
                         lambda i, j: poly.coeff_monomial(self.lmz.var1 ** i * self.lmz.var2 ** j))

    # --- equality (proportional, hashable) -------------------------------
    def _key(self):
        return sp.srepr(self.lmz.canonical_form())

    def __eq__(self, other):
        return isinstance(other, AlgebraicMeasure) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return f"AlgebraicMeasure({self.lmz.expr})"
