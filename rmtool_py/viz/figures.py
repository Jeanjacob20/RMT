"""Laloux figures: spectral density vs MP (Fig. 1), eigenvector components vs
Porter–Thomas (Fig. 2).  Each returns ``(fig, ax)`` and never calls ``show()``.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..finance.spectrum import mp_edges, _mp_density
from ..finance.eigenvectors import component_distribution, porter_thomas_pdf


def plot_spectrum(eigs, *, Q=None, sigma2=1.0, bins=50, ax=None):
    """Histogram of eigenvalues; overlay the closed-form MP density if ``Q`` given."""
    eigs = np.asarray(eigs, dtype=float)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.hist(eigs, bins=bins, density=True, alpha=0.6,
            color="steelblue", label="empirical")
    if Q is not None:
        lo, hi = mp_edges(Q, sigma2)
        grid = np.linspace(max(lo, 1e-6), hi, 400)
        ax.plot(grid, _mp_density(grid, Q, sigma2), "r-", lw=2,
                label="Marčenko–Pastur")
    ax.set_xlabel("eigenvalue λ")
    ax.set_ylabel("ρ(λ)")
    ax.legend()
    return fig, ax


def plot_eigenvector_distribution(eigvec, *, bins=50, ax=None):
    """Histogram of rescaled components u = √N·v vs the Porter–Thomas curve."""
    u = component_distribution(eigvec)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.hist(u, bins=bins, density=True, alpha=0.6,
            color="steelblue", label="components")
    grid = np.linspace(u.min() - 0.5, u.max() + 0.5, 400)
    ax.plot(grid, porter_thomas_pdf(grid), "r-", lw=2, label="Porter–Thomas")
    ax.set_xlabel("u = √N · v")
    ax.set_ylabel("P(u)")
    ax.legend()
    return fig, ax
