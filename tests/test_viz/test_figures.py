import matplotlib
matplotlib.use("Agg")                       # headless, before pyplot import elsewhere

import numpy as np
from matplotlib.figure import Figure

from rmtool_py.viz.figures import plot_spectrum, plot_eigenvector_distribution


def test_plot_spectrum_returns_fig_ax_with_overlay():
    rng = np.random.default_rng(0)
    eigs = rng.gamma(2.0, 0.5, size=400)
    fig, ax = plot_spectrum(eigs, Q=4.0, sigma2=1.0, bins=40)
    assert isinstance(fig, Figure)
    assert len(ax.patches) > 0              # histogram bars
    assert len(ax.lines) >= 1              # MP overlay curve


def test_plot_spectrum_without_overlay():
    rng = np.random.default_rng(1)
    eigs = rng.gamma(2.0, 0.5, size=200)
    fig, ax = plot_spectrum(eigs, bins=30)  # no Q -> no overlay
    assert isinstance(fig, Figure)
    assert len(ax.patches) > 0


def test_plot_eigenvector_distribution():
    rng = np.random.default_rng(2)
    v = rng.standard_normal(256); v = v / np.linalg.norm(v)
    fig, ax = plot_eigenvector_distribution(v, bins=30)
    assert isinstance(fig, Figure)
    assert len(ax.patches) > 0              # component histogram
    assert len(ax.lines) >= 1              # Porter-Thomas curve
