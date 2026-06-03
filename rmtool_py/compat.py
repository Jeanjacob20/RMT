"""RMTool-exact function names (Users Guide v1.0 Sec 2.2) over the Phase-2 core.

Each function takes/returns a BivariatePolynomial in (m, z), matching RMTool's
convention that the user manipulates the bivariate polynomial directly. These are
documented aliases, not new behaviour.
"""

import sympy as sp

from .core import atoms, operations, extract
from .core.measure import AlgebraicMeasure

__all__ = [
    "wignerpol", "wishartpol",
    "invA", "shiftA", "scaleA", "mobiusA", "transposeA", "squareA",
    "AtimesWish", "AgramWish", "AplusB", "AtimesB", "compressA",
    "Lmz2pdf", "Lmz2MomS", "Lmz2MomF",
    "pretty", "latex", "TLmz",
]


# --- measure constructors -------------------------------------------------
def wignerpol():
    """Semicircle / Wigner L_mz (RMTool ``wignerpol``)."""
    return atoms.wigner_lmz()


def wishartpol(c):
    """Marcenko-Pastur / Wishart L_mz with ratio ``c`` (RMTool ``wishartpol``)."""
    return atoms.marchenko_pastur(c)


# --- deterministic laws ---------------------------------------------------
def invA(L):
    return operations.inverse(L)


def shiftA(L, alpha):
    return operations.shift(L, alpha)


def scaleA(L, alpha):
    return operations.scale(L, alpha)


def mobiusA(L, p, q, r, s):
    return operations.mobius(L, p, q, r, s)


def transposeA(L, c):
    return operations.transpose(L, c)


def squareA(L):
    return operations.square(L)


# --- stochastic laws ------------------------------------------------------
def AtimesWish(L, c):
    return operations.times_wishart(L, c)


def AgramWish(L, c, s):
    return operations.gram_wishart(L, c, s)


def AplusB(La, Lb):
    return operations.add(La, Lb)


def AtimesB(La, Lb):
    return operations.mult(La, Lb)


def compressA(La, c):
    return operations.compress(La, c)


# --- extraction -----------------------------------------------------------
def Lmz2pdf(L, xx):
    return extract.density(L, xx)


def Lmz2MomS(L, n):
    return extract.moments(L, n, method="series")


def Lmz2MomF(L, n):
    return extract.moments(L, n, method="fast")


# --- presentation ---------------------------------------------------------
def pretty(L):
    return sp.pretty(L.expr)


def latex(L):
    return sp.latex(L.expr)


def TLmz(L):
    """Matrix of coefficients of L_mz (RMTool ``TLmz``)."""
    return AlgebraicMeasure(L).coefficient_table()
