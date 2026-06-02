# RMTool-Py: Polynomial Engine + Empirical RMT Layer — Design Spec

**Date:** 2026-06-01
**Author:** Jean Jacob (with Claude)
**Status:** Approved design — ready for implementation planning

---

## 1. Purpose & scope

Reproduce N. Raj Rao & Alan Edelman's **RMTool** (a MATLAB Symbolic Toolbox random-matrix "calculator")
as a modern, Pythonic package, and make the empirical results of Laloux–Cizeau–Bouchaud–Potters,
*"Noise Dressing of Financial Correlation Matrices"* (1998), reproducible on top of it.

**V1 deliverable:** a faithful, complete port of RMTool's polynomial method (the engine + the full
operation catalog of the accompanying paper, *The Polynomial Method for Random Matrices*, Rao &
Edelman, Found. Comput. Math. 2008), built around a Pythonic object-oriented core, **plus** the
empirical pieces needed to reproduce Laloux Figures 1 & 2.

**Goal framing** (decided during brainstorming):

- **Modernized re-imagining**, not a literal transliteration: the real API is an OO core; RMTool's
  exact function names are preserved as a thin compatibility shim so existing RMTool demos/papers port
  verbatim.
- **Engine exactness strategy:** layered. `AlgebraicMeasure` keeps an *exact symbolic* bivariate
  polynomial as source of truth; the *numeric* fast-path lives in extraction (root-finding, series).
  "Symbolic by default, numeric where it matters."

**In scope (V1):** the polynomial engine; the RMTool operation catalog; density & moment extraction;
RMTool-name compatibility layer; an empirical/finance module (build correlation matrix, empirical
spectrum, MP fit, eigenvector statistics, market-mode removal); a factor-model **forward** prediction
path (`population ⊠ Marčenko–Pastur`); plotting helpers reproducing Laloux Figs 1 & 2; a real test
suite.

**Out of scope (future / V2+):** RMT-based correlation-matrix **cleaning**/shrinkage (Ledoit–Wolf);
free multiplicative **deconvolution** (the *inverse* problem: recover population spectrum from sample
spectrum); eigenvector/eigenvalue estimation for spiked models. The engine is designed so these slot
in as later layers without rework — the forward free-convolution machinery they depend on is in V1.

### Why the polynomial engine, and not just the closed-form MP formula?

Plain Laloux uses the Marčenko–Pastur law, which is closed form — the engine is optional there. The
engine becomes **necessary** the moment the population correlation is not the identity, e.g. a factor
model: the correct sample-spectrum prediction is then the **free multiplicative convolution of the
population spectrum with Marčenko–Pastur**, which is exactly the engine's `times_wishart(c)` /
`population.mult(MP)` operational law. Factor models produce finite/atomic (hence *algebraic*)
population spectra — precisely the class the polynomial method represents exactly — so this is
tractable. This case retroactively justifies building the engine.

---

## 2. Background: the mathematics we are porting

A measure's limiting eigenvalue distribution `F` is encoded by a **bivariate polynomial** `L_mz(m, z)`
whose zero set defines the Stieltjes transform `m(z) = ∫ 1/(x − z) dF(x)` (Definition 1 of the paper).
The method rests on three layers of machinery (paper §3–4):

1. **Six interconnected encodings** of the same measure (paper Fig. 3, Table 2):
   - `L_mz` — Stieltjes transform `m(z)`
   - `L_gz` — Cauchy transform `g(z) = −m(z)`
   - `L_rg` — R-transform `r(g)`
   - `L_sy` — S-transform `s(y)`
   - `L_μz` — moment transform `μ(z)`
   - `L_ηz` — eta transform `η(z)`
   Conversions between them (paper Table 3, labels I–VI) are variable substitutions followed by an
   irreducibility normalization (`irreducLuv`: clear denominators, make square-free).

2. **Two binary operators on algebraic functions** (paper §4): `⊞` (values add) and `⊠` (values
   multiply), implemented via companion-matrix Kronecker products (Tables 4–5) or via resultants /
   Sylvester matrices (§4.2).

3. **Operational laws** (paper §5–6, Tables 7–9): each random-matrix operation maps one encoding to
   another. The trick: each operation is *simple* in the right encoding — free additive convolution is
   `⊞` in the **R-transform** encoding (`r`'s add); free multiplicative convolution is `⊠` in the
   **S-transform** encoding (`s`'s multiply); deterministic operations are direct substitutions on
   `L_mz`.

Reference polynomials we will reproduce exactly as tests (paper Table 2, Eqs. 1.2–1.4):
- Semicircle / Wigner: `L_mz = m² + zm + 1` (paper Table 2(c) — verified)
- Marčenko–Pastur: the parameter-`c` polynomial of paper Table 2(b) — exact coefficients transcribed
  directly from the table **under test** (do not hand-copy; the build asserts agreement with the
  closed-form MP edges below)
- Worked free sum: `L_mz^{A+B} = m³ + (z+2)m² − (−2z+1)m + 2` (paper Eq. 1.2 — verified)
- Worked free product: `L_mz^{AB} = m⁴z² − 2m³z + m² + 4mz + 4` (paper Eq. 1.4 — verified)

The Laloux MP density and edges (their Eq. 3): with `Q = T/N`, `c = 1/Q`,
`ρ(λ) = (Q / 2πσ²) · √((λ₊ − λ)(λ − λ₋)) / λ`, `λ± = σ²(1 + 1/Q ± 2√(1/Q))`.

---

## 3. Architecture & package layout

Three layers, bottom-up: **core engine** → **RMTool compat shim** → **finance/empirical layer**, with
supporting samplers and visualization.

```
rmtool_py/
  __init__.py           # exposes AlgebraicMeasure, finance, viz, version
  core/
    polynomial.py       # BivariatePolynomial: SymPy poly in 2 vars + irreducLuv normalization
    encodings.py        # six encodings + Table-3 conversion graph (m,g,r,s,μ,η)
    algebra.py          # ⊞ and ⊠ : companion-matrix/Kronecker + resultant binary ops
    atoms.py            # wigner(), marchenko_pastur(c), atomic(weights, points)
    operations.py       # deterministic + stochastic operational laws (Tables 7–9)
    extract.py          # density (roots of L_mz), moments (series / D-finite recursion)
    measure.py          # AlgebraicMeasure  <-- the real OO API (facade over the above)
  compat.py             # RMTool-exact function names -> core
  finance/
    correlation.py      # returns -> C; normalization, market-mode removal
    spectrum.py         # empirical density, MP fit (σ², Q, edges λ±), information eigenvalues
    eigenvectors.py     # Porter–Thomas comparison, inverse participation ratio
    data.py             # factor_model_returns(...) synthetic generator; (optional) returns loaders
  matrices/
    random_matrices.py  # existing samplers (kept, cleaned, seedable)
  viz/
    figures.py          # plot_spectrum (Fig.1), plot_eigenvector_distribution (Fig.2)
tests/                  # real pytest suite (replaces the demo script)
docs/
```

### Data flow — headline use case (Laloux reproduction)

```
returns (N×T) ─► finance.correlation_matrix ─► C ─► finance.spectrum (empirical density + MP fit)
                                                       └─overlay─ core: marchenko_pastur(1/Q).scale(σ²).density(grid)
                                                                   └─► viz.plot_spectrum ─► Fig. 1
C ─► eigen-decomposition ─► finance.eigenvectors (Porter–Thomas, IPR) ─► viz.plot_eigenvector_distribution ─► Fig. 2
```

### Data flow — factor-model forward prediction

```
known population spectrum ─► core.AlgebraicMeasure.atomic(...) ─► .times_wishart(c=N/T)  [ = pop ⊠ MP ]
                                                                   └─► .density(grid) ─► predicted SAMPLE spectrum
simulated factor returns ─► finance.correlation_matrix ─► empirical sample spectrum
  └─ end-to-end validation test: empirical ≈ predicted (within finite-N tolerance)
```

### The OO core API at a glance

```python
from rmtool_py import AlgebraicMeasure as AM

mp        = AM.marchenko_pastur(c=1/3.22).scale(0.74)   # theoretical MP, σ²-scaled
x, dens   = mp.density(grid)                            # density curve
mu        = mp.moments(10)                              # first 10 moments
free_sum  = AM.wigner() + mp                            # free additive convolution
free_prod = AM.wigner() * mp                            # free multiplicative convolution
pop       = AM.atomic(weights=[...], points=[...])      # factor-model population spectrum
sample    = pop.times_wishart(c=N/T)                    # deformed-MP sample-spectrum prediction
```

---

## 4. Component designs

### 4.1 `core.polynomial.BivariatePolynomial`

Foundational type wrapping a SymPy expression in two named variables (e.g. `(m, z)`). Carries the
equivalence-class normalization (`irreducLuv`, paper Table 1):

- `clear_denominators()` — `numden`: multiply out, keep the numerator.
- `make_squarefree()` — divide by `gcd(L, ∂L/∂u)`.
- `simplify()` / `expand()`.

**Invariant:** every operation returns a normalized polynomial, so two encodings of the same measure
compare equal. This is the single most important correctness invariant; it gets dedicated tests.

### 4.2 `core.encodings`

Implements the six encodings (Table 2) and the Table-3 conversions (labels I–VI) as a **transformation
graph**. Each edge = variable substitution + `irreducLuv`. Converting between any two encodings =
shortest path through the graph:
`L_mz ⇄ L_gz ⇄ L_rg`, `L_mz ⇄ L_sy`, `L_mz ⇄ L_μz ⇄ L_ηz`.

### 4.3 `core.algebra`

Two operators on algebraic functions:

- `boxplus(L1, L2)` (⊞) — function whose values are sums of the inputs' values.
- `boxtimes(L1, L2)` (⊠) — products.

Two interchangeable implementations behind one signature, used to cross-check each other in tests:

- **Companion-matrix / Kronecker** (Tables 4–5): companion matrix `Cu` of each poly in `u`; then
  `⊞ → det(uI − ((Cu₁ ⊗ I) + (I ⊗ Cu₂)))`, `⊠ → det(uI − (Cu₁ ⊗ Cu₂))`.
- **Resultant** (§4.2): eliminate the shared variable via the Sylvester-matrix resultant.

Default: resultants (faster, fewer symbols). Companion-matrix path retained as a verification oracle.

### 4.4 `core.atoms`

Constructors returning `AlgebraicMeasure`:

- `wigner()` → `m² + zm + 1`
- `marchenko_pastur(c)` → Table 2(b) polynomial in parameter `c` (coefficients transcribed under test;
  see §2)
- `atomic(weights, points)` → measure `Σ wᵢ δ(x − pᵢ)`, encoded via its Stieltjes transform
  `m(z) = Σ wᵢ/(pᵢ − z)` then `numden` (this is the factor-model population-spectrum constructor).

### 4.5 `core.operations` — the RMTool catalog (paper Tables 7–9)

- **Deterministic** (direct substitution on `L_mz`, no `⊞`/`⊠`):
  `shift(α)` = `A + αI`; `scale(α)` = `αA`; `inverse` = `A⁻¹`; `mobius(p,q,r,s)` = `(pA+qI)/(rA+sI)`;
  `transpose(c)` = `X'X` from `XX'` with size ratio `c`; `square` = `A²`.
- **Stochastic / free:**
  `add` (free additive conv.) = to `L_rg` → `⊞` → back to `L_mz` (R-transforms add);
  `mult` (free multiplicative conv.) = to `L_sy` → `⊠` → back to `L_mz` (S-transforms multiply);
  `times_wishart(c)`, `gram_wishart(c, s)`, `compress(c)`.

The exact substitution formulas are transcribed from the paper's Tables 7–9 **under test** — each has a
worked example in the paper that becomes a regression test (Eqs. 1.2 and 1.4).

### 4.6 `core.extract`

- **Density** (`Lmz2pdf`): for each `x` on the grid, substitute `z = x` (with a small `+iε` offset for
  stability near edges), solve the univariate polynomial in `m` for **all** roots (NumPy `roots`;
  mpmath for hard cases). Density `= (1/π)·Im m` of the selected "right root". Return a `pdfinfo`-style
  structure: `range`, `density`, all roots, and diagnostics.
  **Right-root selection:** default heuristic = root with positive imaginary part and correct
  `m(z) ~ −1/z` asymptotics; expose a `root_selector=` hook and always return all roots so the choice is
  transparent, never silent. Unambiguous for MP (tested against closed form).
- **Moments** (`Lmz2MomS/F`): two paths —
  - *Series* (`MomS`): Laurent/Puiseux expansion of `m(z) = −Σ μ_k z^{−(k+1)}` via SymPy.
  - *D-finite recursion* (`MomF`): finite-depth linear recursion with polynomial coefficients
    (holonomic / `gfun` analog); `O(k)` per moment. Use SymPy's `holonomic` tooling where it fits,
    else implement the recursion directly from `L_mz`.
  The two paths must agree on low-order moments (test), and match closed-form Wigner/MP moments.

### 4.7 `core.measure.AlgebraicMeasure` — the OO facade (real API)

Holds the canonical exact symbolic `L_mz` (other encodings computed lazily). Exposes operations as
methods and Python operators:

| Operator / method | Operation |
|---|---|
| `a + b` | free additive convolution (`add`) |
| `a * b` | free multiplicative convolution (`mult`) |
| `~a` / `a.inverse()` | `A⁻¹` |
| `a.scale(α)`, `a.shift(α)` | `αA`, `A + αI` |
| `a.mobius(p,q,r,s)`, `a.square()`, `a.transpose(c)` | deterministic ops |
| `a.times_wishart(c)`, `a.gram_wishart(c,s)`, `a.compress(c)` | stochastic ops |
| `a.density(grid, **kw)`, `a.moments(k, **kw)` | extraction |
| `a.latex()`, `a.pretty()`, `a.coefficient_table()` | presentation (`latex`, `pretty`, `TLmz`) |

`density()`/`moments()` accept `precision=`/`backend=` to opt into mpmath when NumPy struggles near
edges — this realizes the layered exact+numeric decision.

### 4.8 `compat.py` — RMTool fidelity

Thin, documented-as-aliases wrappers over `core`, matching the user-guide names exactly:
`wignerpol`, `wishartpol(c)`, `invA`, `shiftA`, `scaleA`, `mobiusA`, `transposeA`, `squareA`,
`atimesWish`, `agramWish`, `aplusB`, `atimesB`, `compressA`, `Lmz2pdf`, `Lmz2MomS`, `Lmz2MomF`,
`pretty`, `latex`, `TLmz`.
**Acceptance test:** the user-guide §1.3 quick-start sequence runs and produces the documented
polynomials and curves. This is how we prove "the actual RMTool package as Raj Rao designed it."

### 4.9 `finance/` — empirical RMT layer (Laloux)

- `correlation.py`
  - `correlation_matrix(returns, *, demean=True, standardize=True)` — builds `C = (1/T) M Mᵀ` from an
    `N×T` array/DataFrame of log-returns (subtract mean, rescale each series to unit variance →
    correlation; covariance optional). Records `N`, `T`, `Q = T/N`.
  - `remove_market_mode(C)` — returns deflated matrix + market eigenpair (the `λ₁ ≫ λ_max` mode); supports
    Laloux's `σ² = 1 − λ_max/N` adjustment.
- `spectrum.py`
  - `empirical_density(C, *, method="hist"|"kde", bins=…)`.
  - `mp_edges(Q, sigma2)` → `λ± = σ²(1 + 1/Q ± 2√(1/Q))` (closed form, cross-checked against
    `core.marchenko_pastur(1/Q).scale(σ²)` support).
  - `fit_marchenko_pastur(C_or_eigs, Q, *, fix_Q=True)` → fits `σ²` (optionally effective `Q`) by both
    Laloux estimators: `σ²=1−λ_max/N` and the least-squares "best fit" (their 0.85 → 0.74). Returns
    `σ²`, edges, bulk fraction of eigenvalues/variance (the "94% / 6%" split).
  - `information_eigenvalues(eigs, λ_max)` → eigenvalues above the upper edge ("signal").
- `eigenvectors.py`
  - `component_distribution(eigvec)` → histogram of `u = v_{α,i}` normalized to `Σ u² = N`.
  - `porter_thomas_pdf(u)` → `(1/√2π) e^{−u²/2}`.
  - `inverse_participation_ratio(eigvec)` → `IPR = Σ_i v_{α,i}⁴` (`≈ 3/N` for delocalized RMT vectors).
- `data.py`
  - `factor_model_returns(N, T, *, factors, loadings, idio_var, seed)` — synthetic generator
    `returns = B·F + ε` (market + sector factors + idiosyncratic noise). Doubles as the Laloux showcase
    and the seed for future Ledoit–Wolf-style structured estimators. Population `Σ` is known, enabling
    the ground-truth validation test (§6).
  - (Optional) a convenience returns loader is explicitly **out of scope for the core package**; the
    module stays data-agnostic — it takes a returns matrix. (Decision: keep downloading to the user.)

### 4.10 `viz/figures.py`

- `plot_spectrum(C, fit)` → Fig. 1: empirical density bars + fitted MP curve (from `core`) + inset with
  the market eigenvalue.
- `plot_eigenvector_distribution(C, which="bulk"|"market")` → Fig. 2: bulk-eigenvector component
  histogram vs Porter–Thomas + inset for the market mode.

### 4.11 `matrices/` — samplers (kept, cleaned)

Keep `generate_goe/gue/wishart/inverse_wishart`. Fixes: resolve the currently-ignored `sigma`
parameter in GOE/GUE (honor it or drop it; align with docstring's semicircle-normalization claim);
add seedable RNG (`np.random.default_rng(seed)`) for reproducible tests.

---

## 5. Reproducing Laloux end-to-end (the user-facing workflow)

```python
import rmtool_py as rmt

# returns: N stocks × T days of log-returns (user-supplied; e.g. S&P 500 1991–96, N=406, T=1309)
C   = rmt.finance.correlation_matrix(returns)        # C = (1/T) M Mᵀ, Q = T/N
fit = rmt.finance.fit_marchenko_pastur(C)            # σ² fit, edges λ±, bulk fraction

rmt.viz.plot_spectrum(C, fit)                        # Fig. 1: density + MP curve + market inset
rmt.viz.plot_eigenvector_distribution(C, which="bulk")   # Fig. 2: components vs Porter–Thomas
```

Produces: `Q`, fitted `σ²` (their 0.85 → 0.74 story), `λ_max`, the market eigenvalue ~25–30× the edge,
the ~94%/6% noise/information split, and bulk eigenvectors matching Porter–Thomas while the market mode
deviates.

**Caveats (documented for the user):** (1) data is user-supplied — exact 1991–96 series needed to match
their precise curve; other periods reproduce the qualitative picture. (2) Finite-N edge-blurring is
real; the `σ²` fit is genuinely a fit (both estimators exposed). (3) Fig. 2 is empirical (eigvecs of
`C` vs RMT null) — the polynomial engine is not involved there.

---

## 6. Testing & validation strategy

Test-first (TDD). The existing demo script (`tests/test_matrices/test_wigner.py`) is replaced by a real
`pytest` suite.

| Layer | Assertions |
|---|---|
| `core.polynomial` | `irreducLuv` idempotent; equal measures → equal normalized polynomials |
| `core.encodings` | round-trips (`L_mz → L_rg → L_mz` = identity); Table-2 reference polys reproduced exactly (Wigner, MP, atomic example) |
| `core.algebra` | `⊞`/`⊠` agree between resultant and companion-matrix implementations |
| `core.operations` | paper's worked examples as regression tests (`L_mz^{A+B}` Eq. 1.2, `L_mz^{AB}` Eq. 1.4) |
| `core.extract` | density & moments match closed-form Wigner/MP; slow (`MomS`) vs fast (`MomF`) moments agree |
| `compat` | user-guide §1.3 sequence reproduces documented output |
| `finance` (null) | on sampled i.i.d. data: empirical density of `C` → MP curve (edge/KS check); bulk eigenvectors → Porter–Thomas; market mode detected above `λ_max` |
| **end-to-end (factor model)** | **simulate factor model with known population `Σ`; empirical sample spectrum ≈ engine `atomic(Σ-spectrum).times_wishart(c)` prediction within finite-N tolerance** — the gold-standard validation; plain MP (`Σ=I`) is its special case |

**Convention pinning** (top reproduction hazard): `c = N/T = 1/Q`, the `(1/T)` normalization,
`σ²`-scaling, correlation vs covariance, real-vs-complex factor of 2. Every convention is pinned by
tests that must agree with *both* the closed-form MP edges and the paper's worked polynomials, or the
build fails.

---

## 7. Dependencies & sequencing

**Runtime:** `numpy`, `scipy`, `matplotlib` (existing) + `sympy` (engine) + `pandas` (returns I/O).
`mpmath` ships with SymPy (high-precision extraction). **Dev:** `pytest`.

**Implementation order** (each a testable milestone):
`polynomial` → `encodings` → `algebra` → `atoms` + `operations` → `extract` → `measure` (OO facade) →
`compat` (validate vs user guide) → `finance` (correlation/spectrum/eigenvectors/data) → `viz` →
end-to-end Laloux + factor-model validation.

---

## 8. Scope boundaries & known limitations

- The engine represents **algebraic** measures only. Finite-factor models (atomic population spectra)
  are algebraic ⇒ fully tractable; a non-algebraic continuous population spectrum cannot be encoded
  exactly (not produced by factor models, so outside the practical concern).
- "Right root" selection for multi-modal (factor) densities is heuristic + user-overridable, never
  silent.
- V1 is **forward** only (population → predicted sample spectrum; diagnosis of empirical data).
  Cleaning/shrinkage and free multiplicative **deconvolution** (inverse problem) are V2 — enabled by,
  but not part of, this deliverable.
