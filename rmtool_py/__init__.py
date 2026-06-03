"""Top-level package for Random Matrix Tools."""

__version__ = "0.1.0"

from .core import AlgebraicMeasure
from . import compat

__all__ = ["AlgebraicMeasure", "compat", "__version__"]
