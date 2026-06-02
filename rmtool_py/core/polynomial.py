"""Bivariate-polynomial type carrying the irreducLuv normalization."""

import sympy as sp


class BivariatePolynomial:
    """A bivariate polynomial L(var1, var2) representing a measure encoding.

    ``var1`` is the *transform* variable (m, g, r, s, mu, eta) — the one we
    eliminate / take square-free part with respect to. ``var2`` is the argument
    variable (z, g, y).
    """

    def __init__(self, expr, var1, var2):
        self.var1 = sp.sympify(var1)
        self.var2 = sp.sympify(var2)
        self.expr = sp.expand(sp.sympify(expr))

    @property
    def vars(self):
        return (self.var1, self.var2)

    def clear_denominators(self):
        """Multiply out denominators, keep the numerator (RMTool ``numden``).

        ``sp.cancel`` first reduces the rational expression to lowest terms so a
        factor shared by numerator and denominator (e.g. an ``(m*z + 1)`` left
        over from a transform substitution) is removed rather than retained.
        """
        num, _den = sp.fraction(sp.cancel(sp.together(self.expr)))
        return BivariatePolynomial(sp.expand(num), self.var1, self.var2)

    def make_squarefree(self, var=None):
        """Divide out repeated/content factors w.r.t. ``var`` (default var1).

        Implements the paper's irreducLuv square-free step exactly:
        ``L / gcd(L, dL/dvar)`` using SymPy's *multivariate* gcd, which also
        strips any spurious factor that does not involve ``var``.
        """
        var = self.var1 if var is None else sp.sympify(var)
        e = sp.expand(self.expr)
        common = sp.gcd(e, sp.diff(e, var))
        quotient = sp.cancel(e / common)
        return BivariatePolynomial(sp.expand(quotient), self.var1, self.var2)

    def normalize(self):
        """Full irreducLuv: clear denominators, then square-free w.r.t. both vars."""
        bp = self.clear_denominators()
        bp = bp.make_squarefree(self.var1)
        bp = bp.make_squarefree(self.var2)
        return bp

    def is_proportional_to(self, other):
        """True iff self.expr == c * other.expr for some nonzero constant c."""
        e1, e2 = sp.expand(self.expr), sp.expand(other.expr)
        if e2 == 0:
            return e1 == 0
        ratio = sp.cancel(e1 / e2)
        return len(ratio.free_symbols) == 0 and ratio != 0

    def __repr__(self):
        return f"BivariatePolynomial({self.expr}, {self.var1}, {self.var2})"
