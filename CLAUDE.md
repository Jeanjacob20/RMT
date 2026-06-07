# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rmtool_py` is a modern Python re-imagining of N. Raj Rao & Alan Edelman's **RMTool** (the MATLAB "polynomial method" random-matrix calculator). It is *not* a literal port: the real API is an object-oriented core (`AlgebraicMeasure`), with an RMTool-exact-name compatibility shim (`compat.py`) layered on top.

The end goal is finance applications — reproducing Laloux–Cizeau–Bouchaud–Potters "Noise Dressing of Financial Correlation Matrices" and supporting factor models via free multiplicative convolution (`measure.times_wishart(c)`). The reference papers live in the repo root as PDFs (`RMTool.pdf`, `The Polynomial Method for Random Matrices.pdf` — Tables 1–9 are the operational-law source — `Lalloux_Bouchaud.pdf`, `Ledoit_wolf_review_JFEc.pdf`).

## Commands

```bash
pip install -e ".[dev]"          # editable install with pytest
python3 -m pytest tests          # full suite (~118 tests)
python3 -m pytest tests/test_core -v
python3 -m pytest tests/test_core/test_operations.py::test_name   # single test
```

- Target runtime is Python 3.8; SymPy 1.13.3. `pytest.ini_options` in `pyproject.toml` sets `testpaths=["tests"]` and `-q`.
- `tests/test_matrices/test_wigner.py` is a Phase-1 numeric demo guarded under `if __name__ == "__main__"` — it is intentionally not collected by pytest (it hangs/sampling-heavy). Keep new long-running demos similarly guarded.

## Architecture: the layered engine

Everything is built on one idea: a measure is carried as an **exact symbolic bivariate polynomial** `L_mz(m, z)` (its defining algebraic relation), and numeric extraction (densities, moments) happens *only at the end*. Source of truth is symbolic; floats are a final projection.

`rmtool_py/core/` holds the engine, in dependency order:

- **`polynomial.py` — `BivariatePolynomial(expr, var1, var2)`.** The base type. `var1` is the *transform* variable (m, g, r, s, mu, eta); `var2` is the *argument* variable (z, g, y). `.normalize()` implements the paper's **irreducLuv** square-free reduction: `clear_denominators` (via `sp.cancel`) then `make_squarefree` = `L / gcd(L, dL/dvar)` over *both* variables. This strips factors independent of a variable, so it returns the irreducible defining polynomial — it is idempotent on its own output but **not proportionality-preserving on arbitrary reducible input**. `canonical_form()` underlies the hashable proportional equality used by `AlgebraicMeasure`.
- **`encodings.py`.** The six encodings (`mz/gz/rg/sy/muz/etaz`) and the Table-3 conversions between them. `to_encoding(L, name)` is the router. Rows I–IV use simultaneous dict subs; rows V–VI (muz/etaz) use *sequential* subs transcribed from the paper's MATLAB code column (z first) — see the convention warnings below.
- **`algebra.py` — `boxplus` / `boxtimes`.** The free-convolution primitives: companion-matrix Kronecker construction (Prop. 4.6). `boxplus` = char poly of `(C1⊗I)+(I⊗C2)`, `boxtimes` = of `C1⊗C2`. An explicit `_kron` is kept deliberately (not `sp.kronecker_product`).
- **`atoms.py`.** Constructors returning `L_mz`: `wigner_lmz()`, `atomic_lmz(weights, points)`, `marchenko_pastur(c)`.
- **`operations.py`.** All operational laws (Rao & Edelman Tables 7–9) on `L_mz`. Deterministic laws (`shift/scale/mobius/transpose/square/inverse`) are direct substitutions. Stochastic laws route through encodings: free **add** goes via `rg` (R-transforms add) + `boxplus`; free **mult** via `sy` (S-transforms multiply) + `boxtimes`; `times_wishart/gram_wishart/compress` per §10.2.
- **`extract.py`.** Numeric endpoints. `density` (`Lmz2pdf`: right-root selection, all-roots, degree-0 warning) and `moments` via two paths — `_moments_series` (RMTool MomS, order-by-order) and `_moments_fast` (MomF, D-finite holonomic recurrence, warns on fallback).
- **`measure.py` — `AlgebraicMeasure`.** The OO facade. Constructors `wigner()/marchenko_pastur(c)/atomic(...)`; operators `a + b` (free additive), `a * b` (free multiplicative), `~a` (inverse); methods mirror `operations.py` + `density`/`moments` + presentation (`latex/pretty/coefficient_table`). Equality is **proportional and hashable** (via `canonical_form`).

`compat.py` exposes RMTool-exact names (`wignerpol`, `wishartpol`, `AtimesWish`, `Lmz2pdf`, `TLmz`, …) as thin documented aliases over the core — no new behaviour.

### The finance & viz layers (Phase 3)

These sit *on top of* the symbolic engine: they are the empirical/practitioner side (real eigenvalues, sampled returns, plots), cross-checked against the engine, which stays the source of truth. Both are exported at the package root (`rmtool_py.finance`, `rmtool_py.viz`).

- **`finance/correlation.py`.** `correlation_matrix(returns, *, demean, standardize)` builds `C = (1/T) M Mᵀ` from an N×T matrix and records `N`, `T`, `Q=T/N` (a `Correlation` dataclass). `standardize=True` (default) → correlation matrix (unit diagonal); `standardize=False` → covariance (so the empirical population equals a built Σ — the gold-standard test relies on this). `remove_market_mode(C)` deflates the top (market) eigenpair and reports Laloux `σ² = 1 − λ_max/N` (a `MarketMode` dataclass).
- **`finance/spectrum.py`.** Closed-form Marčenko–Pastur, **convention-pinned to the engine** (`core.marchenko_pastur(1/Q).scale(σ²)`): `mp_edges(Q, σ²)`, `mp_density(λ, Q, σ²)` (public; valid for Q≥1 / c≤1 — it omits the `(1−Q)` atom at 0 for Q<1), `empirical_density` (hist/KDE), `fit_marchenko_pastur` (both Laloux estimators: `1−λ_max/N` and the least-squares bulk fit → an `MPFit`), `information_eigenvalues` (above the upper edge).
- **`finance/eigenvectors.py`.** Empirical eigenvector statistics for Laloux Fig. 2 (the engine does NOT produce these): `component_distribution` (u = √N·v), `porter_thomas_pdf` (standard-normal reference), `inverse_participation_ratio` (Σvᵢ⁴ ≈ 3/N when delocalized).
- **`finance/data.py`.** `factor_model_returns(N, T, *, loadings, factor_cov, idio_var, seed)` = `B·F + ε` with a **known population** `Σ = B Σ_F Bᵀ + diag(idio_var)`, returning `(returns, Σ_pop)`. `loadings=None` → pure idiosyncratic `Σ = diag(idio_var)`, used to build low-cardinality atomic spectra for the gold-standard test.
- **`viz/figures.py`.** Laloux Fig. 1 (`plot_spectrum`: eigenvalue histogram + optional MP overlay) and Fig. 2 (`plot_eigenvector_distribution`: rescaled components + Porter–Thomas). Both return `(fig, ax)` and **never** call `plt.show()`; tests set the Agg backend.

`matrices/random_matrices.py` holds the seedable ensemble samplers (GOE/GUE/Wishart/inverse-Wishart; `np.random.default_rng`, `sigma`/`scale` honored, semicircle radius 2σ). `transforms/`, `deconvolution/`, `utils/` remain placeholders for future work (RMT cleaning/shrinkage, free deconvolution — V2).

## Conventions & reproduction hazards

These are load-bearing and easy to get subtly wrong. Verify any change against both the closed-form values and the paper polynomials.

- **`make_squarefree` must early-return unchanged when `diff(expr, var) == 0`** — otherwise `gcd(e, 0) == e` collapses the polynomial to `1`.
- **muz/etaz conversions use the Table-3 MATLAB *code column* (sequential subs), NOT the printed headers.** `mz→etaz` is `subs(z, -1/z)` then `subs(m, +z*eta)` even though the header prints `m=-z·eta` (a typo). `mz→muz` is `subs(z, 1/z)` then `subs(m, -mu*z)`.
- **Known paper typo, Table 2(b):** the printed `Lηz = czη²+(−zc+1−z)η−1` is wrong; the correct sign is `czη²+(1+z−cz)η−1` (`+z`). The muz cell as printed is correct.
- **Marčenko–Pastur convention pinning is the #1 reproduction hazard.** The engine's ratio is `c = N/T = 1/Q`; MP edges are `(1±√c)²` (engine) vs Laloux's `λ± = σ²(1 + 1/Q ± 2√(1/Q))`. Always pin c-vs-Q, the `1/T` normalization, σ²-scaling, and correlation-vs-covariance before trusting a finance result.

### Verified ground-truth (use as test anchors)

- Wigner/semicircle across encodings (Table 2c): `mz: m²+mz+1`, `gz: g²−gz+1`, `rg: r−g`, `sy: s²y−1`, `muz: μ²z²−μ+1`, `etaz: z²η²−η+1`.
- Atomic at {0,1} weights {½,½} (Eq 3.7): `m(2z²−2z)−(1−2z)`.
- Free additive worked example (Eq 1.2): `m³+(z+2)m²−(−2z+1)m+2`; free mult (Eq 1.4): `m⁴z²−2m³z+m²+4mz+4`.
- Free sum of two unit semicircles → variance-2 semicircle `2m²+mz+1`.
- Moments: MomS and MomF both give Catalan numbers (Wigner) and Narayana (MP).

## Working in this repo

- This is a paper-faithful library: when adding an operation, cite the paper Table/Equation in the docstring (the existing code does this consistently) and add a test that pins it to a closed-form or paper-polynomial value.
- Development has followed subagent-driven TDD with spec + quality review per task; plans and the design spec live under `docs/superpowers/{plans,specs}/`.
- Phases 1 & 2 are complete and merged to `main`. Phase 3 (`finance/`, `viz/`, seedable `matrices/` samplers) is complete on `feature/phase3-finance` (PR #2); its gold-standard test is the headline acceptance criterion: simulate a factor model with known Σ and confirm the empirical sample (covariance) spectrum ≈ engine `atomic(Σ-spectrum).times_wishart(N/T)` (KS on the eigenvalue CDF), with `Σ=I` collapsing to plain MP both symbolically and empirically.
