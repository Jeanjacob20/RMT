# RMTool-Py Phase 3 — Finance / Empirical Layer + Viz Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the practitioner-facing finance/empirical RMT layer (`finance/`), Laloux figures (`viz/`), and clean seedable matrix samplers on top of the Phase-1/2 polynomial engine, validated by a gold-standard factor-model test where the empirical sample spectrum matches the engine's `atomic(Σ-spectrum).times_wishart(N/T)` prediction.

**Architecture:** Three new pieces sit on top of the existing `rmtool_py.core` engine: `finance/` (correlation matrices, empirical spectrum + MP fit, eigenvector statistics, a synthetic factor-model generator), `viz/` (matplotlib Figs 1 & 2, returning `(fig, ax)`, never auto-showing), and a cleaned `matrices/random_matrices.py` (seedable RNG, honored `sigma`). The finance layer keeps a fast closed-form MP density for fitting/plotting; the engine (`core.marchenko_pastur(1/Q).scale(σ²)`, `core.AlgebraicMeasure.atomic(...).times_wishart(c)`) is the source of truth that convention-pinning and gold-standard tests cross-check against.

**Tech Stack:** Python 3.8, numpy, scipy, matplotlib (Agg in tests), sympy (engine, already built), pytest.

**Branch:** `feature/phase3-finance` (already created off `main`; PR #1 / Phase 2 is merged).

**Conventions pinned throughout (the #1 reproduction hazard):** engine ratio `c = N/T = 1/Q` where `Q = T/N`; correlation `C = (1/T) M Mᵀ` with the `1/T` normalization; MP edges `λ± = σ²(1 + 1/Q ± 2√(1/Q))`; `.scale(σ²)` multiplies eigenvalues by `σ²`. The gold-standard test uses **covariance** (`standardize=False`) so the population matches the built `Σ`; null/Laloux tests use the **correlation** default.

---

## File Structure

- `rmtool_py/matrices/random_matrices.py` — MODIFY: seedable RNG, honor `sigma`/`scale`.
- `rmtool_py/finance/__init__.py` — CREATE: re-export the finance API.
- `rmtool_py/finance/correlation.py` — CREATE: `correlation_matrix`, `remove_market_mode` (+ `Correlation`, `MarketMode` dataclasses).
- `rmtool_py/finance/spectrum.py` — CREATE: `mp_edges`, `_mp_density`, `empirical_density`, `fit_marchenko_pastur`, `information_eigenvalues` (+ `MPFit` dataclass).
- `rmtool_py/finance/eigenvectors.py` — CREATE: `component_distribution`, `porter_thomas_pdf`, `inverse_participation_ratio`.
- `rmtool_py/finance/data.py` — CREATE: `factor_model_returns`.
- `rmtool_py/viz/__init__.py` — CREATE: re-export viz API.
- `rmtool_py/viz/figures.py` — CREATE: `plot_spectrum`, `plot_eigenvector_distribution`.
- `rmtool_py/__init__.py` — MODIFY: expose `finance`, `viz`.
- `tests/test_matrices/test_random_matrices.py` — CREATE.
- `tests/test_finance/__init__.py`, `tests/test_finance/test_correlation.py`, `test_spectrum.py`, `test_eigenvectors.py`, `test_data.py`, `test_gold_standard.py`, `test_null_integration.py` — CREATE.
- `tests/test_viz/__init__.py`, `tests/test_viz/test_figures.py` — CREATE.

---

## Task 1: Clean & seed the matrix samplers (honor `sigma`)

**Files:**
- Modify: `rmtool_py/matrices/random_matrices.py`
- Test: `tests/test_matrices/test_random_matrices.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matrices/test_random_matrices.py`:

```python
import numpy as np
import pytest

from rmtool_py.matrices.random_matrices import RandomMatrixGenerator as RMG


def test_goe_reproducible_with_seed():
    a = RMG.generate_goe(40, seed=7)
    b = RMG.generate_goe(40, seed=7)
    assert np.allclose(a, b)


def test_goe_symmetric():
    a = RMG.generate_goe(30, seed=1)
    assert np.allclose(a, a.T)


def test_goe_sigma_scales_linearly():
    # Same seed => same underlying draw; sigma multiplies the matrix.
    base = RMG.generate_goe(30, sigma=1.0, seed=2)
    scaled = RMG.generate_goe(30, sigma=3.0, seed=2)
    assert np.allclose(scaled, 3.0 * base)


def test_goe_semicircle_edge_is_2sigma():
    # Largest eigenvalue of the normalized GOE -> 2*sigma (semicircle radius).
    sigma = 1.5
    a = RMG.generate_goe(800, sigma=sigma, seed=3)
    lam_max = np.linalg.eigvalsh(a).max()
    assert abs(lam_max - 2 * sigma) < 0.2 * sigma


def test_gue_hermitian_and_seeded():
    a = RMG.generate_gue(30, seed=4)
    b = RMG.generate_gue(30, seed=4)
    assert np.allclose(a, b)
    assert np.allclose(a, a.conj().T)


def test_wishart_reproducible_and_normalized():
    a = RMG.generate_wishart(20, 60, seed=5)
    b = RMG.generate_wishart(20, 60, seed=5)
    assert np.allclose(a, b)
    assert a.shape == (20, 20)
    assert np.allclose(a, a.T)


def test_wishart_df_guard():
    with pytest.raises(ValueError):
        RMG.generate_wishart(20, 10, seed=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_matrices/test_random_matrices.py -q`
Expected: FAIL — `generate_goe()` got an unexpected keyword argument `seed` (and `sigma` currently ignored).

- [ ] **Step 3: Rewrite `rmtool_py/matrices/random_matrices.py`**

Replace the whole file with:

```python
"""Random-matrix ensemble samplers (seedable; ``sigma``/``scale`` honored).

GOE/GUE are normalized by sqrt(p) so the limiting spectral law is the semicircle
on [-2, 2]; multiplying by ``sigma`` rescales it to radius 2*sigma (variance
sigma^2), consistent with the engine's ``.scale(sigma**2)``.
"""

import numpy as np
from scipy.stats import wishart, invwishart


class RandomMatrixGenerator:
    """Generators for GOE, GUE, Wishart and inverse-Wishart ensembles.

    Every method accepts ``seed`` (int or ``np.random.Generator``) for
    reproducibility via ``np.random.default_rng``.
    """

    @staticmethod
    def _rng(seed):
        return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    @staticmethod
    def generate_goe(p: int, sigma: float = 1.0, seed=None) -> np.ndarray:
        """p x p Gaussian Orthogonal Ensemble; limiting semicircle on [-2*sigma, 2*sigma]."""
        rng = RandomMatrixGenerator._rng(seed)
        A = np.zeros((p, p))
        idx_lower = np.tril_indices(p, k=-1)
        A[idx_lower] = rng.standard_normal(len(idx_lower[0]))
        A = A + A.T
        diag = rng.standard_normal(p) * np.sqrt(2.0)
        A[np.diag_indices(p)] = diag
        return sigma * (A / np.sqrt(p))

    @staticmethod
    def generate_gue(p: int, sigma: float = 1.0, seed=None) -> np.ndarray:
        """p x p Gaussian Unitary Ensemble; limiting semicircle on [-2*sigma, 2*sigma]."""
        rng = RandomMatrixGenerator._rng(seed)
        idx_lower = np.tril_indices(p, k=-1)
        nlow = len(idx_lower[0])
        re = np.zeros((p, p))
        im = np.zeros((p, p))
        re[idx_lower] = rng.standard_normal(nlow) / np.sqrt(2.0)
        im[idx_lower] = rng.standard_normal(nlow) / np.sqrt(2.0)
        re = re + re.T
        im = im - im.T
        re[np.diag_indices(p)] = rng.standard_normal(p)
        A = re + 1j * im
        return sigma * (A / np.sqrt(p))

    @staticmethod
    def generate_wishart(p: int, n: int, scale: float = 1.0, seed=None) -> np.ndarray:
        """p x p sample covariance W/n, W ~ Wishart(n, scale*I); requires n >= p."""
        if n < p:
            raise ValueError("Degrees of freedom `n` must be >= dimension `p`.")
        rng = RandomMatrixGenerator._rng(seed)
        W = wishart.rvs(df=n, scale=scale * np.eye(p), random_state=rng)
        return W / n

    @staticmethod
    def generate_inverse_wishart(p: int, df: int, scale: float = 1.0, seed=None) -> np.ndarray:
        """p x p inverse-Wishart draw with scale matrix scale*I; requires df >= p."""
        if df < p:
            raise ValueError("Degrees of freedom `df` must be >= dimension `p`.")
        rng = RandomMatrixGenerator._rng(seed)
        return invwishart.rvs(df=df, scale=scale * np.eye(p), random_state=rng)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_matrices/test_random_matrices.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/matrices/random_matrices.py tests/test_matrices/test_random_matrices.py
git commit -m "refactor(matrices): seedable RNG, honor sigma/scale in samplers"
```

---

## Task 2: `finance/correlation.py` — correlation matrix + market-mode removal

**Files:**
- Create: `rmtool_py/finance/__init__.py`
- Create: `rmtool_py/finance/correlation.py`
- Create: `tests/test_finance/__init__.py`
- Test: `tests/test_finance/test_correlation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance/__init__.py` (empty) and `tests/test_finance/test_correlation.py`:

```python
import numpy as np

from rmtool_py.finance.correlation import correlation_matrix, remove_market_mode


def test_shape_and_metadata():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((10, 50))
    out = correlation_matrix(R)
    assert out.C.shape == (10, 10)
    assert out.N == 10 and out.T == 50
    assert out.Q == 50 / 10


def test_correlation_default_unit_diagonal():
    rng = np.random.default_rng(1)
    R = rng.standard_normal((8, 200)) * 5.0 + 3.0  # nonzero mean & variance
    out = correlation_matrix(R)                     # standardize=True default
    assert np.allclose(np.diag(out.C), 1.0, atol=1e-12)
    assert np.allclose(out.C, out.C.T)


def test_covariance_when_not_standardized():
    rng = np.random.default_rng(2)
    R = rng.standard_normal((6, 5000)) * 2.0
    out = correlation_matrix(R, standardize=False)
    # diagonal ~ per-row variance ~ 4.0 (sigma^2)
    assert np.allclose(np.diag(out.C), 4.0, rtol=0.1)


def test_psd():
    rng = np.random.default_rng(3)
    R = rng.standard_normal((12, 100))
    out = correlation_matrix(R)
    assert np.linalg.eigvalsh(out.C).min() > -1e-8


def test_remove_market_mode_deflates_top_eigenvalue():
    rng = np.random.default_rng(4)
    N, T = 50, 400
    common = rng.standard_normal(T)               # a strong common factor
    R = 3.0 * np.outer(np.ones(N), common) + rng.standard_normal((N, T))
    C = correlation_matrix(R).C
    mm = remove_market_mode(C)
    lam_before = np.linalg.eigvalsh(C).max()
    lam_after = np.linalg.eigvalsh(mm.deflated).max()
    assert mm.eigval > lam_after            # market eigenvalue was the top one
    assert lam_after < lam_before
    assert np.isclose(mm.sigma2_residual, 1.0 - mm.eigval / N)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_finance/test_correlation.py -q`
Expected: FAIL — `ModuleNotFoundError: rmtool_py.finance.correlation`.

- [ ] **Step 3: Create `rmtool_py/finance/__init__.py`**

```python
"""Finance / empirical RMT layer (Laloux reproduction)."""

from .correlation import correlation_matrix, remove_market_mode, Correlation, MarketMode

__all__ = ["correlation_matrix", "remove_market_mode", "Correlation", "MarketMode"]
```

- [ ] **Step 4: Create `rmtool_py/finance/correlation.py`**

```python
"""Build correlation/covariance matrices from returns and remove the market mode.

Convention: ``C = (1/T) M Mᵀ`` from an N×T returns matrix (N assets, T obs).
``standardize=True`` (default) rescales each demeaned row to unit variance, so C
is a correlation matrix with unit diagonal; ``standardize=False`` returns the
(1/T-normalized) covariance.  Records N, T and Q = T/N.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Correlation:
    C: np.ndarray
    N: int
    T: int
    Q: float


@dataclass
class MarketMode:
    deflated: np.ndarray     # C with the market mode removed
    eigval: float            # market (largest) eigenvalue
    eigvec: np.ndarray       # market eigenvector
    sigma2_residual: float   # Laloux 1 - lambda_max / N


def correlation_matrix(returns, *, demean=True, standardize=True):
    """Return a :class:`Correlation` built as ``C = (1/T) M Mᵀ`` from N×T returns."""
    M = np.asarray(returns, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns must be a 2-D N×T array")
    N, T = M.shape
    if demean:
        M = M - M.mean(axis=1, keepdims=True)
    if standardize:
        std = np.sqrt((M ** 2).mean(axis=1, keepdims=True))   # population (1/T) std
        std[std == 0] = 1.0
        M = M / std
    C = (M @ M.T) / T
    return Correlation(C=C, N=N, T=T, Q=T / N)


def remove_market_mode(C):
    """Deflate the largest (market) eigenpair; return a :class:`MarketMode`."""
    C = np.asarray(C, dtype=float)
    N = C.shape[0]
    vals, vecs = np.linalg.eigh(C)            # ascending
    lam = vals[-1]
    v = vecs[:, -1]
    deflated = C - lam * np.outer(v, v)
    return MarketMode(deflated=deflated, eigval=lam, eigvec=v,
                      sigma2_residual=1.0 - lam / N)
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_finance/test_correlation.py -q`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add rmtool_py/finance/__init__.py rmtool_py/finance/correlation.py \
        tests/test_finance/__init__.py tests/test_finance/test_correlation.py
git commit -m "feat(finance): correlation_matrix + remove_market_mode"
```

---

## Task 3: `finance/spectrum.py` — MP edges & closed-form density (convention pinning)

**Files:**
- Create: `rmtool_py/finance/spectrum.py`
- Test: `tests/test_finance/test_spectrum.py`

This is the **convention-pinning** task: `mp_edges` and the closed-form `_mp_density` must agree with the engine (`core.marchenko_pastur(1/Q).scale(σ²)`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance/test_spectrum.py`:

```python
import numpy as np

from rmtool_py import AlgebraicMeasure as AM
from rmtool_py.finance.spectrum import mp_edges, _mp_density


def test_mp_edges_closed_form_exact():
    # Q=4, sigma2=1 -> 1/Q=0.25, edges (1+0.25 -/+ 1) = (0.25, 2.25)
    lo, hi = mp_edges(4.0, 1.0)
    assert np.isclose(lo, 0.25)
    assert np.isclose(hi, 2.25)


def test_mp_edges_match_engine_support():
    Q, s2 = 10 / 3.0, 0.74
    lo, hi = mp_edges(Q, s2)
    mp = AM.marchenko_pastur(1.0 / Q).scale(s2)
    grid = np.linspace(1e-3, hi * 1.4, 1500)
    dens = np.array(mp.density(grid).density)
    support = grid[dens > 1e-4]
    assert abs(support.min() - lo) < 0.02 * hi
    assert abs(support.max() - hi) < 0.02 * hi


def test_mp_density_integrates_to_one():
    Q, s2 = 3.0, 1.0
    lo, hi = mp_edges(Q, s2)
    grid = np.linspace(lo, hi, 4000)
    mass = np.trapz(_mp_density(grid, Q, s2), grid)
    assert abs(mass - 1.0) < 1e-2


def test_mp_density_matches_engine_shape():
    # closed-form finance density vs engine density: small KS over the bulk
    Q, s2 = 4.0, 1.0
    lo, hi = mp_edges(Q, s2)
    grid = np.linspace(lo + 1e-3, hi - 1e-3, 400)
    eng = np.array(AM.marchenko_pastur(1.0 / Q).scale(s2).density(grid).density)
    clo = _mp_density(grid, Q, s2)
    # compare normalized cumulative curves (CDF) -> KS-like distance
    def cdf(d):
        c = np.concatenate([[0], np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(grid))])
        return c / c[-1]
    ks = np.max(np.abs(cdf(eng) - cdf(clo)))
    assert ks < 0.02
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_finance/test_spectrum.py -q`
Expected: FAIL — `ImportError: cannot import name 'mp_edges'`.

- [ ] **Step 3: Create `rmtool_py/finance/spectrum.py`**

```python
"""Empirical spectrum, Marčenko–Pastur fit, and information eigenvalues (Laloux).

Closed-form MP (their Eq. 3), with c = N/T = 1/Q:
    edges  λ± = σ²(1 + 1/Q ± 2√(1/Q))
    ρ(λ)   = (Q / 2πσ²) · √((λ₊ − λ)(λ − λ₋)) / λ   on [λ₋, λ₊], else 0.
These are pinned by tests to agree with core.marchenko_pastur(1/Q).scale(σ²).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class MPFit:
    sigma2_market: float        # Laloux 1 - lambda_max/N
    sigma2_lsq: float           # least-squares best fit over the bulk
    edges: tuple                # (lambda_-, lambda_+) at sigma2_lsq
    bulk_eig_fraction: float    # fraction of eigenvalues inside [lambda_-, lambda_+]
    bulk_var_fraction: float    # fraction of total variance inside the bulk


def mp_edges(Q, sigma2=1.0):
    """Marčenko–Pastur support edges (λ₋, λ₊) for ratio c = 1/Q, scale σ²."""
    r = 1.0 / Q
    s = np.sqrt(r)
    return sigma2 * (1.0 + r - 2.0 * s), sigma2 * (1.0 + r + 2.0 * s)


def _mp_density(lam, Q, sigma2=1.0):
    """Closed-form MP density on the grid ``lam`` (0 outside the support)."""
    lam = np.asarray(lam, dtype=float)
    lo, hi = mp_edges(Q, sigma2)
    out = np.zeros_like(lam)
    inside = (lam > lo) & (lam < hi)
    l = lam[inside]
    out[inside] = (Q / (2.0 * np.pi * sigma2)) * np.sqrt((hi - l) * (l - lo)) / l
    return out


def empirical_density(eigs, *, method="hist", bins=50):
    """Empirical spectral density: returns (centers, heights).

    method="hist" -> normalized histogram; method="kde" -> Gaussian KDE sampled
    on a grid spanning the eigenvalues.
    """
    eigs = np.asarray(eigs, dtype=float)
    if method == "hist":
        heights, edges = np.histogram(eigs, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, heights
    if method == "kde":
        from scipy.stats import gaussian_kde
        centers = np.linspace(eigs.min(), eigs.max(), bins)
        return centers, gaussian_kde(eigs)(centers)
    raise ValueError("method must be 'hist' or 'kde'")


def fit_marchenko_pastur(eigs, Q):
    """Fit σ² to the eigenvalue bulk by both Laloux estimators -> :class:`MPFit`.

    Assumes ``eigs`` are eigenvalues of a correlation matrix (trace = N), so the
    market estimator σ² = 1 − λ_max/N is meaningful.
    """
    eigs = np.sort(np.asarray(eigs, dtype=float))
    N = len(eigs)
    sigma2_market = 1.0 - eigs[-1] / N

    def loss(s2):
        lo, hi = mp_edges(Q, s2)
        bulk = eigs[(eigs >= lo) & (eigs <= hi)]
        if len(bulk) < 10:
            return 1e6
        heights, edges = np.histogram(bulk, bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return float(np.mean((heights - _mp_density(centers, Q, s2)) ** 2))

    res = minimize_scalar(loss, bounds=(0.05, 3.0), method="bounded")
    sigma2_lsq = float(res.x)
    lo, hi = mp_edges(Q, sigma2_lsq)
    mask = (eigs >= lo) & (eigs <= hi)
    return MPFit(
        sigma2_market=sigma2_market,
        sigma2_lsq=sigma2_lsq,
        edges=(lo, hi),
        bulk_eig_fraction=float(mask.mean()),
        bulk_var_fraction=float(eigs[mask].sum() / eigs.sum()),
    )


def information_eigenvalues(eigs, lambda_plus):
    """Eigenvalues above the upper MP edge ('signal')."""
    eigs = np.asarray(eigs, dtype=float)
    return eigs[eigs > lambda_plus]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_finance/test_spectrum.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/finance/spectrum.py tests/test_finance/test_spectrum.py
git commit -m "feat(finance): mp_edges + closed-form MP density, convention-pinned to engine"
```

---

## Task 4: `finance/spectrum.py` — fit & information eigenvalues (behavioral tests)

**Files:**
- Modify: `rmtool_py/finance/__init__.py`
- Test: `tests/test_finance/test_spectrum_fit.py`

(`spectrum.py` already contains the functions; this task adds behavioral tests and exports them.)

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance/test_spectrum_fit.py`:

```python
import numpy as np

from rmtool_py.finance.correlation import correlation_matrix
from rmtool_py.finance.spectrum import (
    fit_marchenko_pastur, information_eigenvalues, mp_edges, empirical_density,
)


def _iid_corr_eigs(N, T, seed):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((N, T))
    C = correlation_matrix(R).C
    return np.linalg.eigvalsh(C)


def test_fit_recovers_unit_sigma2_on_null():
    N, T = 400, 1600              # Q=4, c=0.25
    eigs = _iid_corr_eigs(N, T, seed=0)
    fit = fit_marchenko_pastur(eigs, Q=T / N)
    assert abs(fit.sigma2_lsq - 1.0) < 0.1
    assert abs(fit.sigma2_market - 1.0) < 0.1
    assert fit.bulk_eig_fraction > 0.95


def test_fit_market_estimator_drops_with_spike():
    rng = np.random.default_rng(1)
    N, T = 200, 800
    common = rng.standard_normal(T)
    R = 2.5 * np.outer(np.ones(N), common) + rng.standard_normal((N, T))
    eigs = np.linalg.eigvalsh(correlation_matrix(R).C)
    fit = fit_marchenko_pastur(eigs, Q=T / N)
    assert fit.sigma2_market < 0.9          # market share removed
    assert fit.bulk_var_fraction < 1.0


def test_information_eigenvalues_above_edge():
    eigs = np.array([0.1, 0.5, 1.0, 1.8, 9.0, 25.0])
    _, hi = mp_edges(Q=4.0, sigma2=1.0)     # hi = 2.25
    info = information_eigenvalues(eigs, hi)
    assert set(np.round(info, 3)) == {9.0, 25.0}


def test_empirical_density_hist_integrates_to_one():
    eigs = _iid_corr_eigs(300, 1200, seed=2)
    centers, heights = empirical_density(eigs, method="hist", bins=50)
    width = centers[1] - centers[0]
    assert abs(np.sum(heights) * width - 1.0) < 0.05
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_finance/test_spectrum_fit.py -q`
Expected: FAIL — these import fine but assert on behavior; run anyway. If `__init__` export missing, the next step covers it. (If it passes already, still do Step 3 to add exports.)

- [ ] **Step 3: Add spectrum exports to `rmtool_py/finance/__init__.py`**

Replace the file contents with:

```python
"""Finance / empirical RMT layer (Laloux reproduction)."""

from .correlation import correlation_matrix, remove_market_mode, Correlation, MarketMode
from .spectrum import (
    mp_edges, empirical_density, fit_marchenko_pastur, information_eigenvalues, MPFit,
)

__all__ = [
    "correlation_matrix", "remove_market_mode", "Correlation", "MarketMode",
    "mp_edges", "empirical_density", "fit_marchenko_pastur",
    "information_eigenvalues", "MPFit",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_finance/test_spectrum_fit.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/finance/__init__.py tests/test_finance/test_spectrum_fit.py
git commit -m "feat(finance): MP fit (both Laloux estimators) + information eigenvalues"
```

---

## Task 5: `finance/eigenvectors.py` — Porter–Thomas & IPR

**Files:**
- Create: `rmtool_py/finance/eigenvectors.py`
- Modify: `rmtool_py/finance/__init__.py`
- Test: `tests/test_finance/test_eigenvectors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance/test_eigenvectors.py`:

```python
import numpy as np

from rmtool_py.finance.eigenvectors import (
    component_distribution, porter_thomas_pdf, inverse_participation_ratio,
)


def test_component_distribution_normalized_to_N():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(64)
    v = v / np.linalg.norm(v)               # unit eigenvector
    u = component_distribution(v)
    assert np.isclose(np.sum(u ** 2), len(v))


def test_porter_thomas_is_standard_normal_pdf():
    assert np.isclose(porter_thomas_pdf(0.0), 1.0 / np.sqrt(2 * np.pi))
    grid = np.linspace(-8, 8, 4000)
    assert abs(np.trapz(porter_thomas_pdf(grid), grid) - 1.0) < 1e-3


def test_ipr_localized_vs_delocalized():
    e1 = np.zeros(100); e1[0] = 1.0
    assert np.isclose(inverse_participation_ratio(e1), 1.0)   # fully localized
    flat = np.ones(100) / np.sqrt(100)
    assert np.isclose(inverse_participation_ratio(flat), 1.0 / 100)  # fully spread


def test_ipr_of_goe_eigenvector_near_3_over_N():
    rng = np.random.default_rng(3)
    N = 500
    A = rng.standard_normal((N, N)); A = (A + A.T) / 2
    _, V = np.linalg.eigh(A)
    iprs = np.array([inverse_participation_ratio(V[:, k]) for k in range(N)])
    assert abs(np.median(iprs) * N - 3.0) < 0.6      # delocalized ~ 3/N
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_finance/test_eigenvectors.py -q`
Expected: FAIL — `ModuleNotFoundError: rmtool_py.finance.eigenvectors`.

- [ ] **Step 3: Create `rmtool_py/finance/eigenvectors.py`**

```python
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
```

- [ ] **Step 4: Add to `rmtool_py/finance/__init__.py`**

Append the eigenvector imports and extend `__all__`. Replace the file with:

```python
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
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_finance/test_eigenvectors.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add rmtool_py/finance/eigenvectors.py rmtool_py/finance/__init__.py \
        tests/test_finance/test_eigenvectors.py
git commit -m "feat(finance): eigenvector stats (component dist, Porter-Thomas, IPR)"
```

---

## Task 6: `finance/data.py` — factor-model returns generator

**Files:**
- Create: `rmtool_py/finance/data.py`
- Modify: `rmtool_py/finance/__init__.py`
- Test: `tests/test_finance/test_data.py`

Full-Σ_F parameterization: `returns = B·F + ε`, `F ~ N(0, Σ_F)` (K×T), `ε ~ N(0, diag(idio_var))`, population `Σ = B Σ_F Bᵀ + diag(idio_var)`. `loadings=None` → pure idiosyncratic (used by the gold-standard atomic spectrum).

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance/test_data.py`:

```python
import numpy as np

from rmtool_py.finance.data import factor_model_returns
from rmtool_py.finance.correlation import correlation_matrix


def test_shapes_and_population_sigma():
    N, T, K = 20, 5000, 3
    rng = np.random.default_rng(0)
    B = rng.standard_normal((N, K))
    R, Sigma = factor_model_returns(N, T, loadings=B, idio_var=0.5, seed=1)
    assert R.shape == (N, T)
    assert Sigma.shape == (N, N)
    assert np.allclose(Sigma, Sigma.T)
    expected = B @ B.T + 0.5 * np.eye(N)        # factor_cov defaults to I
    assert np.allclose(Sigma, expected)


def test_reproducible_with_seed():
    R1, S1 = factor_model_returns(10, 100, loadings=None, idio_var=1.0, seed=7)
    R2, S2 = factor_model_returns(10, 100, loadings=None, idio_var=1.0, seed=7)
    assert np.allclose(R1, R2) and np.allclose(S1, S2)


def test_pure_idiosyncratic_diagonal_sigma():
    idio = np.array([1.0, 1.0, 3.0, 3.0])
    R, Sigma = factor_model_returns(4, 50, loadings=None, idio_var=idio, seed=2)
    assert np.allclose(Sigma, np.diag(idio))


def test_empirical_covariance_converges_to_population():
    N, T, K = 15, 40000, 2
    rng = np.random.default_rng(3)
    B = rng.standard_normal((N, K))
    Sigma_F = np.array([[2.0, 0.3], [0.3, 1.0]])
    R, Sigma = factor_model_returns(N, T, loadings=B, factor_cov=Sigma_F,
                                    idio_var=0.7, seed=4)
    emp = correlation_matrix(R, standardize=False).C
    assert np.allclose(emp, Sigma, atol=0.15)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_finance/test_data.py -q`
Expected: FAIL — `ModuleNotFoundError: rmtool_py.finance.data`.

- [ ] **Step 3: Create `rmtool_py/finance/data.py`**

```python
"""Synthetic factor-model returns with a KNOWN population covariance.

    returns = B · F + ε,   F ~ N(0, Σ_F) (K×T),   ε_i ~ N(0, idio_var_i)
    population  Σ = B Σ_F Bᵀ + diag(idio_var)

Returns ``(returns, Sigma_pop)`` so callers/tests have the exact population Σ.
With ``loadings=None`` the model is pure idiosyncratic, giving Σ = diag(idio_var)
— used by the gold-standard test to build a low-cardinality atomic spectrum.
"""

import numpy as np


def factor_model_returns(N, T, *, loadings=None, factor_cov=None, idio_var, seed=None):
    """Generate ``(returns N×T, Sigma_pop N×N)`` from a linear factor model."""
    rng = np.random.default_rng(seed)

    idio = np.asarray(idio_var, dtype=float)
    if idio.ndim == 0:
        idio = np.full(N, float(idio))
    if idio.shape != (N,):
        raise ValueError("idio_var must be a scalar or length-N array")

    eps = rng.standard_normal((N, T)) * np.sqrt(idio)[:, None]

    if loadings is None:
        returns = eps
        Sigma = np.diag(idio)
        return returns, Sigma

    B = np.asarray(loadings, dtype=float)
    if B.ndim != 2 or B.shape[0] != N:
        raise ValueError("loadings must be an N×K array")
    K = B.shape[1]

    Sigma_F = np.eye(K) if factor_cov is None else np.asarray(factor_cov, dtype=float)
    if Sigma_F.shape != (K, K):
        raise ValueError("factor_cov must be K×K matching loadings' K")

    L = np.linalg.cholesky(Sigma_F)
    F = L @ rng.standard_normal((K, T))
    returns = B @ F + eps
    Sigma = B @ Sigma_F @ B.T + np.diag(idio)
    return returns, Sigma
```

- [ ] **Step 4: Add to `rmtool_py/finance/__init__.py`**

Add `from .data import factor_model_returns` and include `"factor_model_returns"` in `__all__`. Replace the file with:

```python
"""Finance / empirical RMT layer (Laloux reproduction)."""

from .correlation import correlation_matrix, remove_market_mode, Correlation, MarketMode
from .spectrum import (
    mp_edges, empirical_density, fit_marchenko_pastur, information_eigenvalues, MPFit,
)
from .eigenvectors import (
    component_distribution, porter_thomas_pdf, inverse_participation_ratio,
)
from .data import factor_model_returns

__all__ = [
    "correlation_matrix", "remove_market_mode", "Correlation", "MarketMode",
    "mp_edges", "empirical_density", "fit_marchenko_pastur",
    "information_eigenvalues", "MPFit",
    "component_distribution", "porter_thomas_pdf", "inverse_participation_ratio",
    "factor_model_returns",
]
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_finance/test_data.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add rmtool_py/finance/data.py rmtool_py/finance/__init__.py \
        tests/test_finance/test_data.py
git commit -m "feat(finance): factor_model_returns generator with known population Sigma"
```

---

## Task 7: GOLD STANDARD — factor model vs engine `times_wishart` prediction

**Files:**
- Test: `tests/test_finance/test_gold_standard.py`

The headline acceptance criterion. Build a population with a known **atomic** spectrum, simulate sample covariance, and assert the empirical eigenvalue distribution matches the engine's `atomic(Σ-spectrum).times_wishart(N/T).density(...)` within finite-N tolerance (measured by KS distance on the eigenvalue CDF). Uses **covariance** (`standardize=False`) so the empirical population is exactly the built `Σ`. Plain MP (Σ=I) is the special case.

Reference values measured during planning (seeded): 2-atom KS ≈ 0.005–0.006, plain-MP KS ≈ 0.006; tolerance 0.05 leaves a ~10× margin while staying a genuine distributional test.

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance/test_gold_standard.py`:

```python
import numpy as np

from rmtool_py import AlgebraicMeasure as AM
from rmtool_py.finance.data import factor_model_returns
from rmtool_py.finance.correlation import correlation_matrix


def _predicted_cdf(measure, xs, gridmax):
    g = np.linspace(1e-4, gridmax, 2000)
    d = np.array(measure.density(g).density)
    cdf = np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(g))])
    cdf /= cdf[-1]
    return np.interp(xs, g, cdf)


def _ks(empirical_sorted, measure, gridmax):
    n = len(empirical_sorted)
    Fpred = _predicted_cdf(measure, empirical_sorted, gridmax)
    Femp = np.arange(1, n + 1) / n
    return float(np.max(np.abs(Femp - Fpred)))


def test_gold_standard_two_atom_population():
    # Population spectrum: half eigenvalues at 1.0, half at 3.0 (weights 1/2, 1/2).
    c = 0.3
    N, T = 400, int(round(400 / c))
    a, b = 1.0, 3.0
    idio = np.array([a] * (N // 2) + [b] * (N // 2))
    R, Sigma = factor_model_returns(N, T, loadings=None, idio_var=idio, seed=0)

    # population spectrum -> atomic measure -> deformed-MP sample prediction
    pop = AM.atomic([0.5, 0.5], [a, b])
    predicted = pop.times_wishart(c)

    eigs = np.sort(np.linalg.eigvalsh(correlation_matrix(R, standardize=False).C))
    ks = _ks(eigs, predicted, gridmax=7.0)
    assert ks < 0.05                      # measured ~0.005


def test_gold_standard_plain_mp_special_case():
    # Sigma = I  ->  prediction collapses to plain Marčenko–Pastur.
    c = 0.25
    N, T = 400, int(round(400 / c))
    R, Sigma = factor_model_returns(N, T, loadings=None, idio_var=1.0, seed=1)
    assert np.allclose(Sigma, np.eye(N))

    predicted = AM.atomic([1.0], [1.0]).times_wishart(c)
    assert predicted == AM.marchenko_pastur(c)        # exact symbolic identity

    eigs = np.sort(np.linalg.eigvalsh(correlation_matrix(R, standardize=False).C))
    ks = _ks(eigs, predicted, gridmax=3.0)
    assert ks < 0.05
```

- [ ] **Step 2: Run to verify it fails (or, if engine present, passes)**

Run: `python3 -m pytest tests/test_finance/test_gold_standard.py -q`
Expected: PASS (the engine and finance pieces already exist). If it FAILS on tolerance, do NOT loosen blindly — invoke `superpowers:systematic-debugging`: re-check the `c = N/T = 1/Q` convention, `standardize=False`, and the atom weights/points before touching the tolerance.

- [ ] **Step 3: Commit**

```bash
git add tests/test_finance/test_gold_standard.py
git commit -m "test(finance): gold-standard factor-model spectrum vs engine times_wishart"
```

---

## Task 8: Null integration — i.i.d. → MP + Porter–Thomas + market detection

**Files:**
- Test: `tests/test_finance/test_null_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance/test_null_integration.py`:

```python
import numpy as np

from rmtool_py import AlgebraicMeasure as AM
from rmtool_py.finance.correlation import correlation_matrix, remove_market_mode
from rmtool_py.finance.spectrum import mp_edges, information_eigenvalues
from rmtool_py.finance.eigenvectors import inverse_participation_ratio


def _predicted_cdf(measure, xs, gridmax):
    g = np.linspace(1e-4, gridmax, 2000)
    d = np.array(measure.density(g).density)
    cdf = np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(g))])
    cdf /= cdf[-1]
    return np.interp(xs, g, cdf)


def test_null_spectrum_matches_mp():
    c = 0.25
    N, T = 400, int(round(400 / c))
    rng = np.random.default_rng(0)
    C = correlation_matrix(rng.standard_normal((N, T))).C
    eigs = np.sort(np.linalg.eigvalsh(C))

    lo, hi = mp_edges(Q=T / N, sigma2=1.0)
    assert abs(eigs.min() - lo) < 0.05
    assert eigs.max() < hi + 0.15           # finite-N edge blur

    mp = AM.marchenko_pastur(c)
    Fpred = _predicted_cdf(mp, eigs, gridmax=3.0)
    ks = np.max(np.abs(np.arange(1, len(eigs) + 1) / len(eigs) - Fpred))
    assert ks < 0.05                        # measured ~0.006


def test_null_bulk_eigenvectors_delocalized():
    rng = np.random.default_rng(1)
    N, T = 300, 1500
    C = correlation_matrix(rng.standard_normal((N, T))).C
    _, V = np.linalg.eigh(C)
    iprs = np.array([inverse_participation_ratio(V[:, k]) for k in range(N)])
    assert abs(np.median(iprs) * N - 3.0) < 0.7


def test_market_mode_detected_and_removed():
    rng = np.random.default_rng(2)
    N, T = 200, 1000
    common = rng.standard_normal(T)
    R = 2.5 * np.outer(np.ones(N), common) + rng.standard_normal((N, T))
    C = correlation_matrix(R).C
    eigs = np.linalg.eigvalsh(C)

    _, hi = mp_edges(Q=T / N, sigma2=1.0)
    info = information_eigenvalues(eigs, hi)
    assert len(info) >= 1                    # market eigenvalue above the edge
    assert info.max() > 5 * hi               # ≫ λ_max, as Laloux report

    mm = remove_market_mode(C)
    assert np.linalg.eigvalsh(mm.deflated).max() < mm.eigval
```

- [ ] **Step 2: Run to verify it passes**

Run: `python3 -m pytest tests/test_finance/test_null_integration.py -q`
Expected: 3 passed. (If a tolerance fails, debug the convention before loosening — see Task 7 Step 2.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_finance/test_null_integration.py
git commit -m "test(finance): null-case integration (MP, Porter-Thomas IPR, market mode)"
```

---

## Task 9: `viz/figures.py` — Laloux Figs 1 & 2

**Files:**
- Create: `rmtool_py/viz/__init__.py`
- Create: `rmtool_py/viz/figures.py`
- Create: `tests/test_viz/__init__.py`
- Test: `tests/test_viz/test_figures.py`

Matplotlib, **Agg backend in tests**, return `(fig, ax)`, never call `plt.show()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_viz/__init__.py` (empty) and `tests/test_viz/test_figures.py`:

```python
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
    assert len(ax.lines) >= 1               # MP overlay curve


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
    assert len(ax.lines) >= 1               # Porter-Thomas curve
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_viz/test_figures.py -q`
Expected: FAIL — `ModuleNotFoundError: rmtool_py.viz.figures`.

- [ ] **Step 3: Create `rmtool_py/viz/__init__.py`**

```python
"""Visualization helpers (Laloux Figs 1 & 2)."""

from .figures import plot_spectrum, plot_eigenvector_distribution

__all__ = ["plot_spectrum", "plot_eigenvector_distribution"]
```

- [ ] **Step 4: Create `rmtool_py/viz/figures.py`**

```python
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
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_viz/test_figures.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add rmtool_py/viz/__init__.py rmtool_py/viz/figures.py \
        tests/test_viz/__init__.py tests/test_viz/test_figures.py
git commit -m "feat(viz): plot_spectrum (Fig.1) + plot_eigenvector_distribution (Fig.2)"
```

---

## Task 10: Wire up package exports & run the full suite

**Files:**
- Modify: `rmtool_py/__init__.py`

- [ ] **Step 1: Read the current `rmtool_py/__init__.py`**

Run: `sed -n '1,60p' rmtool_py/__init__.py` (read it first — do not clobber existing `AlgebraicMeasure`/`compat`/`matrices` exports).

- [ ] **Step 2: Expose `finance` and `viz`**

Add these imports near the other subpackage imports (adjust to match the file's existing style), and extend `__all__`:

```python
from . import finance, viz

# extend (do not clobber) any existing __all__
try:
    __all__  # noqa: F821
except NameError:
    __all__ = []
__all__ += ["finance", "viz"]
```

- [ ] **Step 3: Verify imports**

Run:
```bash
python3 -c "import rmtool_py as rmt; from rmtool_py import finance, viz; print(rmt.finance.correlation_matrix, rmt.viz.plot_spectrum, rmt.finance.factor_model_returns); print('imports ok')"
```
Expected: prints the three callables and `imports ok`, no error.

- [ ] **Step 4: Run the entire suite**

Run: `python3 -m pytest tests -q`
Expected: all green. Phase-1/2 ≈ 77 + Phase-3 new (matrices 7, correlation 5, spectrum 4, spectrum_fit 4, eigenvectors 4, data 4, gold_standard 2, null_integration 3, viz 3 = 36) ≈ **113 passed**, zero failures. (Exact count may vary by ±a few; zero failures is the bar.)

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/__init__.py
git commit -m "feat: export finance + viz from package root"
```

---

## Phase 3 completion criteria

- [ ] `python3 -m pytest tests -q` → all green, no skips of ground-truth tests.
- [ ] `python3 -c "from rmtool_py import finance, viz"` → no error.
- [ ] **Convention pinning:** `mp_edges(Q, σ²)` agrees with closed form AND with `core.marchenko_pastur(1/Q).scale(σ²)` support; closed-form `_mp_density` matches engine density (KS < 0.02).
- [ ] **Null case:** i.i.d. correlation spectrum → MP (edges + KS < 0.05); bulk eigenvector IPR ≈ 3/N; planted market mode detected above λ_max and removable.
- [ ] **GOLD STANDARD:** simulated 2-atom factor model → empirical covariance spectrum matches `atomic([½,½],[a,b]).times_wishart(N/T)` (KS < 0.05); Σ=I special case equals plain MP both symbolically and empirically.
- [ ] `correlation_matrix` correlation default (unit diagonal) + covariance option; `factor_model_returns` returns known population Σ.
- [ ] `viz` functions return `(fig, ax)`, render histogram + overlay, never auto-show.
- [ ] `matrices` samplers seedable; `sigma` honored (semicircle radius 2σ).

---

## Self-review notes (spec coverage)

- Spec §4.9 correlation.py → Task 2 ✓; spectrum.py → Tasks 3–4 ✓; eigenvectors.py → Task 5 ✓; data.py → Task 6 ✓.
- Spec §4.10 viz → Task 9 ✓. Spec §4.11 matrices → Task 1 ✓.
- Spec §6 testing table: convention pinning (Task 3), null finance (Task 8), end-to-end factor model (Task 7) ✓.
- Decisions baked in: correlation default (Task 2), histogram default + KS-on-eigenvalues (Tasks 4/7/8), full-Σ_F factor model (Task 6), honor `sigma` (Task 1), `(fig, ax)` return (Task 9), gold standard on covariance (Task 7).
- Deferred to V2 (not in this plan, per spec §8): RMT cleaning/shrinkage, free multiplicative deconvolution, returns loaders.
```