"""Density and moment extraction from the L_mz bivariate polynomial.

Moments use the moment transform mu(z) = sum_j M_j z^j (M_0 = 1), which satisfies
L_muz(mu, z) = 0. The "series" path (RMTool MomS) solves order-by-order; the "fast"
path (RMTool MomF) builds a D-finite linear recurrence and unrolls it.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.holonomic.holonomicerrors import BaseHolonomicError

from . import encodings as enc

z, mu = sp.symbols("z mu")


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
    except (NotImplementedError, ValueError, BaseHolonomicError):
        warnings.warn(
            "_moments_fast: holonomic path failed; falling back to series method",
            stacklevel=2,
        )
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


@dataclass
class PdfInfo:
    """Result of density extraction (RMTool ``pdfinfo``)."""
    range: list        # grid x-values
    density: list      # (1/pi) Im m at the selected right root
    all_roots: list    # list of all m-roots per grid point (transparency)


def _default_right_root(roots):
    """Physical Stieltjes branch: among Im>0 roots, the largest imaginary part.

    This is the default heuristic (largest Im among upper-half-plane roots),
    not a guaranteed analytic continuation for every spectral law.
    """
    upper = [r for r in roots if r.imag > 1e-9]
    if not upper:
        return 0.0 + 0.0j
    return max(upper, key=lambda r: r.imag)


def density(L, grid, eps=1e-8, root_selector=None):
    """Limiting eigenvalue density on ``grid`` (RMTool ``Lmz2pdf``).

    For each x, solve L_mz(m, x + i*eps) for all roots in m and report
    rho(x) = (1/pi) Im m_right.  All roots are returned in ``all_roots``
    for transparency; ``root_selector(roots) -> complex`` overrides the
    default branch choice.

    Root-selector heuristic: the default picks the upper-half-plane root
    with the largest imaginary part.  This is unambiguous for single-branch
    laws (Wigner, Marchenko-Pastur), but may select the wrong analytic
    continuation for multi-branch laws (e.g. free sum of two MPs).  Pass a
    custom ``root_selector`` or inspect ``all_roots`` in those cases.

    eps trade-off: larger eps regularises near-real roots (smooth density
    estimate) but smears sharp edges; smaller eps is more accurate in the
    bulk but may admit numerical noise from roots that are nearly real.
    """
    select = root_selector or _default_right_root
    mvar = L.var1
    poly_m = sp.Poly(L.expr, mvar)
    zvar = L.var2
    rng, dens, allroots = [], [], []
    for x in grid:
        zc = complex(x) + 1j * eps
        coeffs = [complex(c.subs(zvar, zc)) for c in poly_m.all_coeffs()]
        if len(coeffs) <= 1:
            warnings.warn(
                f"density: L_mz polynomial is degree 0 in m at x={x!r} "
                "(constant in m); this indicates a degenerate or mis-encoded "
                "spectral law — density reported as 0 at this point.",
                stacklevel=2,
            )
            roots = []
        else:
            roots = list(np.roots(coeffs))
        rng.append(float(x))
        allroots.append(roots)
        dens.append(max(select(roots).imag / np.pi, 0.0))
    return PdfInfo(range=rng, density=dens, all_roots=allroots)
