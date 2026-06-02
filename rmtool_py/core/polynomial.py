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
        if self.expr == 0:
            return BivariatePolynomial(sp.Integer(0), self.var1, self.var2)
        e = sp.expand(self.expr)
        deriv = sp.diff(e, var)
        if deriv == 0:
            # expr is independent of var: no repeated factors in var to remove,
            # and gcd(e, 0) == e would otherwise collapse e to the constant 1.
            return BivariatePolynomial(e, self.var1, self.var2)
        common = sp.gcd(e, deriv)
        quotient = sp.cancel(e / common)
        return BivariatePolynomial(sp.expand(quotient), self.var1, self.var2)

    def normalize(self):
        """Full irreducLuv (Rao & Edelman 2008, Table 1): clear denominators,
        then take the square-free part w.r.t. BOTH variables.

        Taking gcd with the derivative w.r.t. each variable removes repeated
        factors and any factor independent of that variable. For an irreducible
        encoding polynomial this is a no-op; for a reducible one it discards
        spurious curve components (e.g. a constant-m branch), returning the
        irreducible defining polynomial. It is therefore NOT proportionality-
        preserving on arbitrary reducible input, but IS idempotent on its own
        (irreducible) output.
        """
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

    def canonical_form(self):
        """A canonical SymPy expression invariant under multiplication by a
        nonzero constant (the equivalence-class representative).

        normalize -> view every free symbol as a polynomial generator (so all
        coefficients are rational numbers) -> remove integer content -> fix the
        leading-term sign positive. Two measures whose L_mz are proportional by a
        nonzero constant share an identical canonical_form (hashable, == comparable).
        """
        e = self.normalize().expr
        if e == 0 or not e.free_symbols:
            return sp.Integer(0) if e == 0 else sp.Integer(1)
        gens = sorted(e.free_symbols, key=lambda sym: sym.sort_key())
        poly = sp.Poly(e, *gens)
        _content, prim = poly.primitive()       # coeffs become coprime integers
        lead_coeff = prim.terms()[0][1]          # leading monomial coeff (a number)
        prim_expr = prim.as_expr()
        if lead_coeff < 0:
            prim_expr = -prim_expr
        return sp.expand(prim_expr)

    def __repr__(self):
        return f"BivariatePolynomial({self.expr}, {self.var1}, {self.var2})"
