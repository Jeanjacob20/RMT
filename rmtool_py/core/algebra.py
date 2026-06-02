"""Binary operators on algebraic functions: boxplus and boxtimes.

Given two bivariate polynomials L1(u,v), L2(u,v) defining algebraic functions
u1(v), u2(v) (as roots in u), boxplus returns the polynomial whose roots are
u1(v)+u2(v) and boxtimes the one whose roots are u1(v)*u2(v) (Prop. 4.6).
"""

import sympy as sp

from .polynomial import BivariatePolynomial


def companion_matrix(bp, u):
    """Companion matrix C_u of bp treated as a polynomial in u (Table 4).

    Char poly det(u I - C_u) equals bp made monic in u.
    """
    u = sp.sympify(u)
    poly = sp.Poly(sp.expand(bp.expr), u)
    coeffs = poly.all_coeffs()           # leading first: [l_Du, ..., l_1, l_0]
    deg = poly.degree()
    lead = coeffs[0]
    C = sp.zeros(deg, deg)
    for i in range(1, deg):
        C[i, i - 1] = 1
    for j in range(deg):                 # coeff of u^j is coeffs[deg - j]
        l_j = coeffs[deg - j]
        C[j, deg - 1] = sp.cancel(-l_j / lead)
    return C


def _kron(A, B):
    ar, ac = A.shape
    br, bc = B.shape
    M = sp.zeros(ar * br, ac * bc)
    for i in range(ar):
        for j in range(ac):
            M[i * br:(i + 1) * br, j * bc:(j + 1) * bc] = A[i, j] * B
    return M


def _charpoly(C, u, other_var):
    n = C.shape[0]
    chi = sp.expand((u * sp.eye(n) - C).det())
    return BivariatePolynomial(chi, u, other_var).normalize()


def _other_var(bp, u):
    others = [s for s in (bp.var1, bp.var2) if s != sp.sympify(u)]
    return others[0]


def boxplus(bp1, bp2, u):
    """u1(v) + u2(v): eigenvalues of (C1 ⊗ I) + (I ⊗ C2) (Prop. 4.6.1)."""
    u = sp.sympify(u)
    other = _other_var(bp1, u)
    C1 = companion_matrix(bp1, u)
    C2 = companion_matrix(bp2, u)
    C3 = _kron(C1, sp.eye(C2.shape[0])) + _kron(sp.eye(C1.shape[0]), C2)
    return _charpoly(C3, u, other)


def boxtimes(bp1, bp2, u):
    """u1(v) * u2(v): eigenvalues of C1 ⊗ C2 (Prop. 4.6.2)."""
    u = sp.sympify(u)
    other = _other_var(bp1, u)
    C1 = companion_matrix(bp1, u)
    C2 = companion_matrix(bp2, u)
    C3 = _kron(C1, C2)
    return _charpoly(C3, u, other)
