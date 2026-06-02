"""Density and moment extraction from the L_mz bivariate polynomial.

Moments use the moment transform mu(z) = sum_j M_j z^j (M_0 = 1), which satisfies
L_muz(mu, z) = 0. The "series" path (RMTool MomS) solves order-by-order; the "fast"
path (RMTool MomF) builds a D-finite linear recurrence and unrolls it.
"""

import sympy as sp

from . import encodings as enc

m, z, mu = sp.symbols("m z mu")


def _moments_series(L, k):
    """MomS: solve for the mu(z) power-series coefficients order-by-order."""
    lmuz = enc.to_encoding(L, "muz").expr
    coeffs = sp.symbols("M1:%d" % (k + 1))             # M1 .. Mk
    series = 1 + sum(coeffs[i] * z ** (i + 1) for i in range(k))
    poly = sp.expand(sp.numer(sp.cancel(sp.together(lmuz.subs(mu, series)))))
    sol, moments = {}, []
    for j in range(1, k + 1):
        cj = sp.expand(poly.coeff(z, j).subs(sol))
        unknown = [sym for sym in cj.free_symbols if sym in coeffs]
        if unknown:
            sol[unknown[0]] = sp.solve(cj, unknown[0])[0]
            moments.append(sp.simplify(sol[unknown[0]]))
        else:
            moments.append(sp.Integer(0))
    return moments


def _moments_fast(L, k):
    """MomF: D-finite linear recurrence for the moment sequence (RMTool fast path).

    Solve L_muz for the branch mu(z) with mu(0) = 1, build its holonomic
    representation, read the recurrence + initial terms, and unroll. Falls back to
    the series method if the holonomic machinery cannot handle this polynomial.
    """
    try:
        from sympy.holonomic import expr_to_holonomic
        lmuz = enc.to_encoding(L, "muz").expr
        branch = None
        for root in sp.solve(lmuz, mu):
            if sp.limit(root, z, 0) == 1:
                branch = root
                break
        if branch is None:
            raise ValueError("no mu(0)=1 branch")
        hol = expr_to_holonomic(branch, z, x0=0)
        ser = hol.series(n=k + 1)                       # uses the recurrence internally
        poly = ser.removeO() if hasattr(ser, "removeO") else ser
        return [sp.nsimplify(poly.coeff(z, j)) for j in range(1, k + 1)]
    except Exception:
        return _moments_series(L, k)


def moments(L, k, method="series"):
    """First ``k`` moments [M_1, ..., M_k] of the measure encoded by ``L`` (M_0 = 1).

    method="series" (MomS, exact order-by-order) or "fast" (MomF, D-finite recurrence).
    """
    if method == "series":
        return _moments_series(L, k)
    if method == "fast":
        return _moments_fast(L, k)
    raise ValueError("method must be 'series' or 'fast'")
