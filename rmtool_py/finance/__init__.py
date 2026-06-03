"""Finance / empirical RMT layer (Laloux reproduction)."""

from .correlation import correlation_matrix, remove_market_mode, Correlation, MarketMode
from .spectrum import (
    mp_edges, empirical_density, fit_marchenko_pastur, information_eigenvalues, MPFit,
)
from .eigenvectors import (
    component_distribution, porter_thomas_pdf, inverse_participation_ratio,
)

__all__ = [
    "correlation_matrix", "remove_market_mode", "Correlation", "MarketMode",
    "mp_edges", "empirical_density", "fit_marchenko_pastur",
    "information_eigenvalues", "MPFit",
    "component_distribution", "porter_thomas_pdf", "inverse_participation_ratio",
]
