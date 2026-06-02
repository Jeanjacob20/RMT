"""Top-level package for the core polynomial-method engine."""

from .polynomial import BivariatePolynomial
from . import encodings, algebra, atoms

__all__ = ["BivariatePolynomial", "encodings", "algebra", "atoms"]
