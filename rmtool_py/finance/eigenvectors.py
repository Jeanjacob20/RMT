"""Eigenvector statistics for the Laloux null comparison (Fig. 2).

For a delocalized RMT eigenvector the rescaled components u = √N · v are
distributed as a standard normal (Porter–Thomas), and IPR = Σ vᵢ⁴ ≈ 3/N.
"""

import numpy as np


def component_distribution(eigvec):
    """Rescale a unit eigenvector so that Σ uᵢ² = N (u = √N · v)."""
    v = np.asarray(eigvec, dtype=float)
    return v * np.sqrt(len(v))


def porter_thomas_pdf(u):
    """Standard-normal density (1/√2π) e^{−u²/2}, the Porter–Thomas reference."""
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)


def inverse_participation_ratio(eigvec):
    """IPR = Σ_i vᵢ⁴ for a unit eigenvector (≈ 3/N when delocalized, 1 when localized)."""
    v = np.asarray(eigvec, dtype=float)
    return float(np.sum(v ** 4))
