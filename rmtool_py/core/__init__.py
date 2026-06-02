"""Core polynomial-method engine (bivariate-polynomial substrate)."""

from .polynomial import BivariatePolynomial

try:
    from . import encodings, algebra, atoms
except ImportError:
    pass

__all__ = ["BivariatePolynomial", "encodings", "algebra", "atoms"]
