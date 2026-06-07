"""Top-level package for Random Matrix Tools."""

__version__ = "0.1.0"

from .core import AlgebraicMeasure
from . import compat, finance, viz

__all__ = ["AlgebraicMeasure", "compat", "finance", "viz", "__version__"]
