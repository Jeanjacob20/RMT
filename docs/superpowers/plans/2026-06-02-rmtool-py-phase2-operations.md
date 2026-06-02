# RMTool-Py Phase 2: Operations, Extraction & OO Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full RMTool polynomial-method calculator on top of the Phase-1 substrate — the moment/eta encodings, the Marčenko–Pastur atom, the deterministic + stochastic operational laws (Tables 7–9), density & moment extraction, the `AlgebraicMeasure` OO facade, and the RMTool-exact compatibility shim — every formula transcribed from the paper and pinned to paper-verified ground-truth values.

**Architecture:** All operations are exact symbolic substitutions / `⊞`/`⊠` compositions on the Phase-1 `BivariatePolynomial` (canonical `L_mz`). The six encodings let each random-matrix operation be applied in whichever transform makes it a simple substitution (free addition in `rg`, free multiplication in `sy`, compression in `rg`). Numerics live only in `extract.py` (NumPy/mpmath root-finding for density; SymPy series + SymPy holonomic recursion for moments). `measure.AlgebraicMeasure` is a thin facade over `operations`/`extract`; `compat.py` is a thin RMTool-named shim over the facade-free core functions.

**Tech Stack:** Python 3.8, SymPy 1.13 (including `sympy.holonomic`), NumPy (density root-finding), mpmath (ships with SymPy; high-precision opt-in), pytest.

**Reference:** Rao & Edelman, *The Polynomial Method for Random Matrices*, Found. Comput. Math. 8 (2008) 649–702 — Table 2 (encodings of reference measures), Table 3 rows IV–VI (S/moment/eta conversions), Tables 7–9 (operational laws), §2.2 (moment & eta transforms), §10.2 (random compression). RMTool Users Guide v1.0 §1.3, §2.2, §2.5 (compat function names + acceptance sequence). Spec: `docs/superpowers/specs/2026-06-01-rmtool-py-polynomial-engine-design.md`. Phase-1 plan: `docs/superpowers/plans/2026-06-01-rmtool-py-phase1-core-foundations.md`.

---

## Paper-verified ground-truth anchors (all reproduced with the Phase-1 engine before this plan was written)

These exact values are used as tests. Every one was computed and confirmed against the paper during planning; do not alter them to make code pass — a mismatch means the code is wrong.

**Encodings (Table 2). `μ`=`mu`, `η`=`eta`.**
- Wigner: `Lμz = μ²z² − μ + 1`, `Lηz = z²η² − η + 1` (Table 2(c)).
- Atomic `½δ₀+½δ₁`: `Lμz = (−2+2z)μ + 2 − z`, `Lηz = (2z+2)η − 2 − z` (Table 2(a)).
- Marčenko–Pastur: `Lμz = czμ² − (zc+1−z)μ + 1` (Table 2(b)), and `Lηz = czη² + (1+z−cz)η − 1`.
  > **CORRECTION (paper typo).** Table 2(b) as printed gives `Lηz = czη² + (−zc+1−z)η − 1`. That is a typo: applying the paper's own row-VI substitution to the MP `Lmz`, *and* evaluating the eta-transform definition Eq. 2.10 `η(z)=(1/z)·m(−1/z)` numerically (e.g. `c=1, z=2 ⇒ η=0.5`), both give the `+z` form `czη² + (1+z−cz)η − 1`. Use the `+z` form. The `−z` printed cell fails the numeric check.

**Table-3 substitutions (transcribed verbatim from the MATLAB code column; applied sequentially as written).**
- IV `Lmz→Lsy`: already in Phase 1.
- V  `Lmz→Lμz`: `subs(z,1/z)` then `subs(m,-mu*z)`. Inverse `Lμz→Lmz`: `subs(z,1/z)` then `subs(mu,-m*z)`.
- VI `Lmz→Lηz`: `subs(z,-1/z)` then `subs(m,z*eta)`. Inverse `Lηz→Lmz`: `subs(z,-1/z)` then `subs(eta,-z*m)`.
  > The paper's row-VI *header* prints `m = −zη`; its MATLAB *code* uses `m = z*eta` (`+`). The `+` form is correct (reproduces Table 2(c) Wigner and Table 2(a) atomic, and round-trips). Trust the code column, not the header.

**Marčenko–Pastur atom.** `Lmz = czm² − (1−c−z)m + 1` (Table 2(b)). Density-support edges = roots of the `m`-discriminant `(1−c−z)² − 4cz = 0`, i.e. `z = (1 ± √c)²`. With Laloux's `c = 1/Q` and `.scale(σ²)`, edges become `σ²(1 + 1/Q ± 2√(1/Q))` (their λ±).

**Operational laws (Tables 7–9, §10.2). All worked examples below were reproduced exactly (or up to a nonzero constant, noted) by the engine.**
- `shift(α)` `A+αI`: `z → z−α`. shift(δ₃,2) ∝ δ₅.
- `scale(α)` `αA`: `m→αm, z→z/α`. scale(δ₃,2) ∝ δ₆.
- `inverse` `A⁻¹` = `mobius(0,1,1,0)`: net `m→−z²m−z, z→1/z`. inverse(δ₃) ∝ δ_{1/3}.
- `mobius(p,q,r,s)` `(pA+qI)/(rA+sI)`: `α=(q−sz)/(p−rz)`, `β=1/(p−rz)`; `subs(z,−α)` then `subs(m,((m/β)−r)/(s−rα))`.
- `square` `A²`: `⊞` over `m` of `L_mz(2m√z, √z)` and `L_mz(−2m√z, −√z)`. square(δ₃) ∝ δ₉.
- `transpose(c)` (`XX'→X'X`, `c=SizeA/SizeB`; project/augment law, Table 7, with `α=0`): `m → (1−1/c)/(0−z) + m/c`. transpose(δ₃, c=½) ∝ `½δ₀ + ½δ₃`.
- `add` (`A⊞B`, free additive, Table 9(a)): both `→rg`, `⊞` over `r`, back `→mz`. add(Wigner, MP(½)) = `m³ + (z+2)m² + (2z−1)m + 2` **(Eq. 1.2, exact)**.
- `mult` (`A⊠B`, free multiplicative, Table 9(b)): both `→sy`, `⊠` over `s`, back `→mz`. mult(Wigner, MP(½)) ∝ `m⁴z² − 2m³z + m² + 4mz + 4` **(Eq. 1.4, factor −1)**.
- `times_wishart(c)` (`A×W(c)`, "Multiply Wishart", Table 7): `α=(1−c−czm)`; `m→mα, z→z/α` (simultaneous). times_wishart(Wigner, ½) ∝ Eq. 1.4 (factor 4); and `times_wishart(L,c) ∝ mult(L, MP(c))`.
- `gram_wishart(c,s)` ("Grammian", Table 7): `α=1+scm`, `β=α(zα+s(c−1))`; `m→m/α, z→β` (simultaneous). gram_wishart(δ₀, c, s) ∝ MP(c).scale(s).
- `compress(c)` (random compression, §10.2 Eq. 10.8): `→rg`, `g→c·g`, back `→mz`. compress(`½δ₀+½δ₁`, c) = `(−2cz²+2cz)m² − (−2c+4cz+1−2z)m + (−2c+2)` **(Eq. 10.9, exact)**.

**Moments (§2.2).** `μ(z)=Σ_{j≥0} M_j z^j`, `M_0=1`, satisfies `L_μz(μ,z)=0`. Wigner moments `M_1..M_7 = 0,1,0,2,0,5,0` (odd=0, even=Catalan). MP(c) moments `M_1..M_4 = 1, 1+c, 1+3c+c², 1+6c+6c²+c³` (Narayana). MP(½) `M_1..M_3 = 1, 1.5, 2.75`.

---

## File structure

```
rmtool_py/
  __init__.py          # MODIFY: expose AlgebraicMeasure, compat, version
  core/
    __init__.py        # MODIFY: export operations, extract, measure, marchenko_pastur
    polynomial.py      # MODIFY: add canonical_form() (proportional-equality key)
    encodings.py       # MODIFY: add muz/etaz encodings + Table-3 rows V/VI
    atoms.py           # MODIFY: add marchenko_pastur(c)
    operations.py      # CREATE: deterministic + stochastic operational laws (Tables 7-9)
    extract.py         # CREATE: density (Lmz2pdf) + moments (series MomS, holonomic MomF)
    measure.py         # CREATE: AlgebraicMeasure OO facade
  compat.py            # CREATE: RMTool-exact function names
tests/
  test_core/
    test_encodings.py  # MODIFY: append muz/etaz tests
    test_atoms.py      # MODIFY: append marchenko_pastur tests
    test_polynomial.py # MODIFY: append canonical_form tests
    test_operations.py # CREATE
    test_extract.py    # CREATE
    test_measure.py    # CREATE
  test_compat.py       # CREATE: §1.3 / §2.5 acceptance
```

Branch: `feature/phase2-operations` off `main`.

---

## Task 0: Branch setup

- [ ] **Step 1: Create the feature branch**

Run:
```bash
cd /Users/jeanjacob/RMT-3
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b feature/phase2-operations
python3 -m pytest tests/test_core -q
```
Expected: branch created; `28 passed` (Phase-1 baseline green before any change).

---

## Task 1: `encodings.py` — moment (`muz`) and eta (`etaz`) encodings

**Files:**
- Modify: `rmtool_py/core/encodings.py`
- Test: `tests/test_core/test_encodings.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_core/test_encodings.py`:

```python
from rmtool_py.core.atoms import atomic_lmz

mu, eta = sp.symbols("mu eta")


def test_wigner_muz_etaz():
    lmz = wigner_lmz()
    assert _prop(enc.to_encoding(lmz, "muz"), mu ** 2 * z ** 2 - mu + 1, mu, z)
    assert _prop(enc.to_encoding(lmz, "etaz"), z ** 2 * eta ** 2 - eta + 1, eta, z)


def test_atomic_muz_etaz():
    # Table 2(a): atoms at 0 and 1, weights 1/2.
    lmz = atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 1])
    assert _prop(enc.to_encoding(lmz, "muz"), (-2 + 2 * z) * mu + 2 - z, mu, z)
    assert _prop(enc.to_encoding(lmz, "etaz"), (2 * z + 2) * eta - 2 - z, eta, z)


def test_mp_muz_etaz():
    # Marchenko-Pastur Table 2(b). NB: the etaz cell uses the corrected (+z) form;
    # the paper's printed (-z) cell is a typo (fails eta-transform Eq. 2.10).
    c = sp.symbols("c")
    lmz = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    assert _prop(enc.to_encoding(lmz, "muz"),
                 c * z * mu ** 2 - (z * c + 1 - z) * mu + 1, mu, z)
    assert _prop(enc.to_encoding(lmz, "etaz"),
                 c * z * eta ** 2 + (1 + z - c * z) * eta - 1, eta, z)


def test_roundtrip_mz_muz_mz():
    c = sp.symbols("c")
    lmz = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    back = enc.to_encoding(enc.to_encoding(lmz, "muz"), "mz")
    assert back.is_proportional_to(lmz)


def test_roundtrip_mz_etaz_mz():
    c = sp.symbols("c")
    lmz = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    back = enc.to_encoding(enc.to_encoding(lmz, "etaz"), "mz")
    assert back.is_proportional_to(lmz)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_encodings.py -k "muz or etaz" -v`
Expected: FAIL — `ValueError: unknown target encoding 'muz'`.

- [ ] **Step 3: Add the two encodings to `encodings.py`**

In `rmtool_py/core/encodings.py`, change the symbols line and `ENCODING_VARS`:

```python
m, z, g, r, s, y, mu, eta = sp.symbols("m z g r s y mu eta")

# encoding name -> (transform var, argument var)
ENCODING_VARS = {
    "mz": (m, z),
    "gz": (g, z),
    "rg": (r, g),
    "sy": (s, y),
    "muz": (mu, z),
    "etaz": (eta, z),
}
```

Add these four conversion functions after `sy_to_mz` (substitutions are applied
**sequentially**, exactly as the paper's Table-3 MATLAB code column):

```python
def mz_to_muz(e):                # V:  Lmuz = Lmz(-mu*z, 1/z)
    return e.subs(z, 1 / z).subs(m, -mu * z)


def muz_to_mz(e):                # V:  Lmz = Lmuz(-m*z, 1/z)
    return e.subs(z, 1 / z).subs(mu, -m * z)


def mz_to_etaz(e):               # VI: Letaz = Lmz(z*eta, -1/z)   (code column: +z*eta)
    return e.subs(z, -1 / z).subs(m, z * eta)


def etaz_to_mz(e):               # VI: Lmz = Letaz(-z*m, -1/z)
    return e.subs(z, -1 / z).subs(eta, -z * m)
```

Add the four edges to `_EDGES`:

```python
_EDGES = {
    ("mz", "gz"): mz_to_gz, ("gz", "mz"): gz_to_mz,
    ("gz", "rg"): gz_to_rg, ("rg", "gz"): rg_to_gz,
    ("mz", "sy"): mz_to_sy, ("sy", "mz"): sy_to_mz,
    ("mz", "muz"): mz_to_muz, ("muz", "mz"): muz_to_mz,
    ("mz", "etaz"): mz_to_etaz, ("etaz", "mz"): etaz_to_mz,
}
```

(The `_ADJ` adjacency and BFS rebuild automatically from `_EDGES`.)

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_encodings.py -v`
Expected: all pass (previous Phase-1 encoding tests + 5 new).

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/encodings.py tests/test_core/test_encodings.py
git commit -m "feat(core): add muz/etaz encodings (Table-3 rows V-VI); fix Table-2b etaz typo"
```

---

## Task 2: `atoms.py` — `marchenko_pastur(c)`

**Files:**
- Modify: `rmtool_py/core/atoms.py`
- Test: `tests/test_core/test_atoms.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_core/test_atoms.py`:

```python
from rmtool_py.core.atoms import marchenko_pastur

c = sp.symbols("c")


def test_marchenko_pastur_table_2b():
    # Table 2(b): L_mz = c z m^2 - (1 - c - z) m + 1
    bp = marchenko_pastur(c)
    expected = BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
    assert bp.is_proportional_to(expected)


def test_marchenko_pastur_edges():
    # Density-support edges are roots of the m-discriminant; must equal (1 +/- sqrt(c))^2.
    cval = sp.Rational(1, 4)
    bp = marchenko_pastur(cval)
    poly_m = sp.Poly(bp.expr, m)
    a, b, cc = poly_m.all_coeffs()            # a m^2 + b m + cc
    disc = sp.expand(b ** 2 - 4 * a * cc)     # quadratic in z
    edges = sorted(sp.solve(disc, z))
    expected = sorted([(1 - sp.sqrt(cval)) ** 2, (1 + sp.sqrt(cval)) ** 2])
    assert [sp.nsimplify(e) for e in edges] == [sp.nsimplify(e) for e in expected]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_atoms.py -k marchenko -v`
Expected: FAIL — `ImportError: cannot import name 'marchenko_pastur'`.

- [ ] **Step 3: Implement `marchenko_pastur`**

Add to `rmtool_py/core/atoms.py` (the module already has `m, z = sp.symbols("m z")`):

```python
def marchenko_pastur(c):
    """Marcenko-Pastur law with aspect ratio ``c`` (Table 2(b)).

    L_mz = c z m^2 - (1 - c - z) m + 1.  The density support is
    [(1 - sqrt(c))^2, (1 + sqrt(c))^2]; ``c`` may be numeric or symbolic.
    For Laloux's normalization use c = N/T = 1/Q and scale by sigma^2.
    """
    c = sp.sympify(c)
    return BivariatePolynomial(c * z * m ** 2 - (1 - c - z) * m + 1, m, z)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_atoms.py -v`
Expected: all pass (Phase-1 atomic/wigner + 2 new).

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/atoms.py tests/test_core/test_atoms.py
git commit -m "feat(core): marchenko_pastur(c) atom, validated against MP edges"
```

---

## Task 3: `polynomial.py` — `canonical_form()` for proportional equality

**Files:**
- Modify: `rmtool_py/core/polynomial.py`
- Test: `tests/test_core/test_polynomial.py` (append)

This is the hashable canonical representative used by `AlgebraicMeasure.__eq__`/`__hash__`:
normalize, treat every free symbol as a polynomial generator (so all coefficients are
rational numbers), strip integer content, fix the leading-term sign.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_core/test_polynomial.py`:

```python
def test_canonical_form_collapses_constant_multiple():
    a = BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert a.canonical_form() == b.canonical_form()


def test_canonical_form_collapses_negative_and_fraction():
    a = BivariatePolynomial(-sp.Rational(2, 3) * (m ** 2 + m * z + 1), m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert a.canonical_form() == b.canonical_form()


def test_canonical_form_distinguishes_measures():
    a = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    c = BivariatePolynomial(m ** 2 + m * z + 2, m, z)
    assert a.canonical_form() != c.canonical_form()


def test_canonical_form_is_hashable_and_consistent():
    a = BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert hash(a.canonical_form()) == hash(b.canonical_form())
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_polynomial.py -k canonical -v`
Expected: FAIL — `AttributeError: 'BivariatePolynomial' object has no attribute 'canonical_form'`.

- [ ] **Step 3: Implement `canonical_form`**

Add this method to `BivariatePolynomial` (after `is_proportional_to`):

```python
    def canonical_form(self):
        """A canonical SymPy expression invariant under multiplication by a
        nonzero constant (the equivalence-class representative).

        normalize -> view every free symbol as a polynomial generator (so all
        coefficients are rational numbers) -> remove integer content -> fix the
        leading-term sign positive. Two measures whose L_mz are proportional by a
        nonzero constant share an identical canonical_form (hashable, == comparable).
        """
        e = self.normalize().expr
        if e == 0 or not e.free_symbols:
            return sp.Integer(0) if e == 0 else sp.Integer(1)
        gens = sorted(e.free_symbols, key=lambda sym: sym.sort_key())
        poly = sp.Poly(e, *gens)
        _content, prim = poly.primitive()       # coeffs become coprime integers
        lead_coeff = prim.terms()[0][1]          # leading monomial coeff (a number)
        prim_expr = prim.as_expr()
        if lead_coeff < 0:
            prim_expr = -prim_expr
        return sp.expand(prim_expr)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_polynomial.py -v`
Expected: all pass (Phase-1 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/polynomial.py tests/test_core/test_polynomial.py
git commit -m "feat(core): BivariatePolynomial.canonical_form (hashable proportional-equality key)"
```

---

## Task 4: `operations.py` — deterministic laws (Table 7, Table 8)

**Files:**
- Create: `rmtool_py/core/operations.py`
- Test: `tests/test_core/test_operations.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core/test_operations.py`:

```python
import sympy as sp

from rmtool_py.core.polynomial import BivariatePolynomial
from rmtool_py.core.atoms import wigner_lmz, atomic_lmz, marchenko_pastur
from rmtool_py.core import operations as ops

m, z, c = sp.symbols("m z c")


def _atom(a):
    return atomic_lmz([1], [a])


def _prop(bp, other):
    return bp.is_proportional_to(other)


def test_shift_atom():
    assert _prop(ops.shift(_atom(3), 2), _atom(5))


def test_scale_atom():
    assert _prop(ops.scale(_atom(3), 2), _atom(6))


def test_inverse_atom():
    assert _prop(ops.inverse(_atom(3)), _atom(sp.Rational(1, 3)))


def test_square_atom():
    assert _prop(ops.square(_atom(3)), _atom(9))


def test_mobius_reduces_to_shift():
    # mobius(1, alpha, 0, 1) == shift(alpha): (1*A + alpha I)/(0*A + I) = A + alpha I
    assert _prop(ops.mobius(_atom(3), 1, 2, 0, 1), ops.shift(_atom(3), 2))


def test_transpose_atom_adds_zero_mass():
    # transpose(delta_3, c=1/2) == 1/2 delta_0 + 1/2 delta_3
    out = ops.transpose(_atom(3), sp.Rational(1, 2))
    expected = atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 3])
    assert _prop(out, expected)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_operations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.core.operations'`.

- [ ] **Step 3: Implement the deterministic laws**

Create `rmtool_py/core/operations.py`:

```python
"""Operational laws on the L_mz bivariate polynomial (Rao & Edelman Tables 7-9).

Each function takes and returns a BivariatePolynomial in (m, z). Deterministic
laws are direct substitutions; stochastic laws (Task 5) route through the rg / sy
encodings and the boxplus / boxtimes operators.
"""

import sympy as sp

from .polynomial import BivariatePolynomial
from . import encodings as enc
from .algebra import boxplus, boxtimes

m, z = sp.symbols("m z")


def _mz(expr):
    """Normalize an expression into an L_mz BivariatePolynomial."""
    return BivariatePolynomial(expr, m, z).normalize()


def shift(L, alpha):
    """A + alpha*I  (Table 7, "Translate"):  z -> z - alpha."""
    alpha = sp.sympify(alpha)
    return _mz(L.expr.subs(z, z - alpha, simultaneous=True))


def scale(L, alpha):
    """alpha*A  (Table 7, "Scale"):  m -> alpha*m, z -> z/alpha."""
    alpha = sp.sympify(alpha)
    return _mz(L.expr.subs({m: alpha * m, z: z / alpha}, simultaneous=True))


def mobius(L, p, q, r, s):
    """(p*A + q*I) / (r*A + s*I)  (Table 7, "Mobius"). Sequential subs as printed:
    alpha = (q - s z)/(p - r z); beta = 1/(p - r z); z -> -alpha; m -> ((m/beta) - r)/(s - r*alpha)."""
    p, q, r, s = map(sp.sympify, (p, q, r, s))
    alpha = (q - s * z) / (p - r * z)
    beta = 1 / (p - r * z)
    e = L.expr.subs(z, -alpha, simultaneous=True)
    e = e.subs(m, ((m / beta) - r) / (s - r * alpha), simultaneous=True)
    return _mz(e)


def inverse(L):
    """A^{-1}  (Table 7, "Invert") = mobius(0, 1, 1, 0)."""
    return mobius(L, 0, 1, 1, 0)


def square(L):
    """A^2  (Table 8(a)): boxplus over m of the two square-root branches."""
    sqz = sp.sqrt(z)
    L1 = _mz(L.expr.subs({z: sqz, m: 2 * m * sqz}, simultaneous=True))
    L2 = _mz(L.expr.subs({z: -sqz, m: -2 * m * sqz}, simultaneous=True))
    return _mz(boxplus(L1, L2, m).expr)


def transpose(L, c):
    """If A = X X' then B = X' X, with c = Size(A)/Size(B)  (Table 7, project/augment
    with the augmenting atom at 0):  m -> (1 - 1/c)/(0 - z) + m/c."""
    c = sp.sympify(c)
    mb = (1 - 1 / c) * (1 / (0 - z)) + m / c
    return _mz(L.expr.subs(m, mb, simultaneous=True))
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_operations.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/operations.py tests/test_core/test_operations.py
git commit -m "feat(core): deterministic operational laws (shift/scale/inverse/mobius/square/transpose)"
```

---

## Task 5: `operations.py` — stochastic laws (Tables 7, 9; §10.2)

**Files:**
- Modify: `rmtool_py/core/operations.py`
- Test: `tests/test_core/test_operations.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_core/test_operations.py`:

```python
def test_add_free_sum_eq_1_2():
    # Eq. 1.2: A = Wigner, B = MP(1/2). L_mz^{A+B} = m^3 + (z+2)m^2 + (2z-1)m + 2.
    out = ops.add(wigner_lmz(), marchenko_pastur(sp.Rational(1, 2)))
    expected = BivariatePolynomial(
        m ** 3 + (z + 2) * m ** 2 + (2 * z - 1) * m + 2, m, z)
    assert _prop(out, expected)


def test_mult_free_product_eq_1_4():
    # Eq. 1.4: L_mz^{AB} = m^4 z^2 - 2 m^3 z + m^2 + 4 m z + 4  (up to a constant).
    out = ops.mult(wigner_lmz(), marchenko_pastur(sp.Rational(1, 2)))
    expected = BivariatePolynomial(
        m ** 4 * z ** 2 - 2 * m ** 3 * z + m ** 2 + 4 * m * z + 4, m, z)
    assert _prop(out, expected)


def test_times_wishart_matches_eq_1_4():
    out = ops.times_wishart(wigner_lmz(), sp.Rational(1, 2))
    expected = BivariatePolynomial(
        m ** 4 * z ** 2 - 2 * m ** 3 * z + m ** 2 + 4 * m * z + 4, m, z)
    assert _prop(out, expected)


def test_times_wishart_equals_mult_with_mp():
    # A x W(c) is the free multiplicative convolution of A with MP(c).
    lhs = ops.times_wishart(wigner_lmz(), sp.Rational(1, 2))
    rhs = ops.mult(wigner_lmz(), marchenko_pastur(sp.Rational(1, 2)))
    assert _prop(lhs, rhs)


def test_gram_wishart_of_zero_atom_is_scaled_mp():
    # (sqrt(A) + sqrt(s) G)(...)' with A = delta_0 is s * W(c) = MP(c).scale(s).
    s = sp.symbols("s")
    out = ops.gram_wishart(atomic_lmz([1], [0]), c, s)
    expected = ops.scale(marchenko_pastur(c), s)
    assert _prop(out, expected)


def test_compress_atomic_eq_10_9():
    # Sec 10.2 worked example: compress(1/2 delta_0 + 1/2 delta_1, c).
    out = ops.compress(atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 1]), c)
    expected = BivariatePolynomial(
        (-2 * c * z ** 2 + 2 * c * z) * m ** 2
        - (-2 * c + 4 * c * z + 1 - 2 * z) * m
        + (-2 * c + 2), m, z)
    assert _prop(out, expected)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_operations.py -k "add or mult or wishart or compress" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'add'`.

- [ ] **Step 3: Implement the stochastic laws**

Append to `rmtool_py/core/operations.py`:

```python
def add(L1, L2):
    """A + B, free additive convolution  (Table 9(a)): rg -> boxplus over r -> mz."""
    r = sp.symbols("r")
    lr1 = enc.to_encoding(L1, "rg")
    lr2 = enc.to_encoding(L2, "rg")
    lrc = boxplus(lr1, lr2, r)
    return enc.to_encoding(lrc, "mz")


def mult(L1, L2):
    """A x B, free multiplicative convolution  (Table 9(b)): sy -> boxtimes over s -> mz."""
    s = sp.symbols("s")
    ls1 = enc.to_encoding(L1, "sy")
    ls2 = enc.to_encoding(L2, "sy")
    lsc = boxtimes(ls1, ls2, s)
    return enc.to_encoding(lsc, "mz")


def times_wishart(L, c):
    """A x W(c), "Multiply Wishart"  (Table 7):
    alpha = 1 - c - c z m; m -> m*alpha, z -> z/alpha  (simultaneous)."""
    c = sp.sympify(c)
    alpha = 1 - c - c * z * m
    return _mz(L.expr.subs({m: m * alpha, z: z / alpha}, simultaneous=True))


def gram_wishart(L, c, s):
    """(sqrt(A) + sqrt(s) G)(sqrt(A) + sqrt(s) G)', "Grammian"  (Table 7):
    alpha = 1 + s c m; beta = alpha*(z alpha + s (c - 1)); m -> m/alpha, z -> beta."""
    c, s = sp.sympify(c), sp.sympify(s)
    alpha = 1 + s * c * m
    beta = alpha * (z * alpha + s * (c - 1))
    return _mz(L.expr.subs({m: m / alpha, z: beta}, simultaneous=True))


def compress(L, c):
    """Random compression by factor c  (Sec 10.2, Eq. 10.8): in rg, g -> c*g."""
    c = sp.sympify(c)
    g = sp.symbols("g")
    lrg = enc.to_encoding(L, "rg")
    compressed = BivariatePolynomial(
        lrg.expr.subs(g, c * g, simultaneous=True), *lrg.vars)
    return enc.to_encoding(compressed, "mz")
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_operations.py -v`
Expected: 13 passed (6 deterministic + 7 stochastic).

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/operations.py tests/test_core/test_operations.py
git commit -m "feat(core): stochastic operational laws (add/mult/times_wishart/gram_wishart/compress)"
```

---

## Task 6: `extract.py` — moments (series `MomS` + holonomic `MomF`)

**Files:**
- Create: `rmtool_py/core/extract.py`
- Test: `tests/test_core/test_extract.py` (create)

`moments` returns the physical moments `[M_1, ..., M_k]` (`M_0 = 1` is implicit).
`method="series"` (MomS) solves order-by-order from `L_μz`; `method="fast"` (MomF)
builds a D-finite linear recurrence from the `μ(z)` branch and unrolls it, falling
back to the series method if the holonomic machinery cannot handle a given polynomial.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core/test_extract.py`:

```python
import sympy as sp

from rmtool_py.core.atoms import wigner_lmz, marchenko_pastur
from rmtool_py.core import extract

c = sp.symbols("c")


def test_moments_series_wigner_catalan():
    # Semicircle: M_1..M_7 = 0,1,0,2,0,5,0.
    got = extract.moments(wigner_lmz(), 7, method="series")
    assert [sp.nsimplify(x) for x in got] == [0, 1, 0, 2, 0, 5, 0]


def test_moments_series_mp_narayana():
    # MP(c): M_1..M_4 = 1, 1+c, 1+3c+c^2, 1+6c+6c^2+c^3.
    got = extract.moments(marchenko_pastur(c), 4, method="series")
    expected = [sp.Integer(1), 1 + c, 1 + 3 * c + c ** 2,
                1 + 6 * c + 6 * c ** 2 + c ** 3]
    assert [sp.expand(x) for x in got] == [sp.expand(e) for e in expected]


def test_moments_fast_matches_series_wigner():
    fast = extract.moments(wigner_lmz(), 7, method="fast")
    slow = extract.moments(wigner_lmz(), 7, method="series")
    assert [sp.nsimplify(a) for a in fast] == [sp.nsimplify(b) for b in slow]


def test_moments_fast_matches_series_mp_numeric():
    mp = marchenko_pastur(sp.Rational(1, 2))
    fast = extract.moments(mp, 6, method="fast")
    slow = extract.moments(mp, 6, method="series")
    assert [sp.nsimplify(a) for a in fast] == [sp.nsimplify(b) for b in slow]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.core.extract'`.

- [ ] **Step 3: Implement the moment extractors**

Create `rmtool_py/core/extract.py`:

```python
"""Density and moment extraction from the L_mz bivariate polynomial.

Moments use the moment transform mu(z) = sum_j M_j z^j (M_0 = 1), which satisfies
L_muz(mu, z) = 0. The "series" path (RMTool MomS) solves order-by-order; the "fast"
path (RMTool MomF) builds a D-finite linear recurrence and unrolls it.
"""

import sympy as sp

from . import encodings as enc

m, z, mu = sp.symbols("m z mu")


def _moments_series(L, k):
    """MomS: solve for the mu(z) power-series coefficients order-by-order."""
    lmuz = enc.to_encoding(L, "muz").expr
    coeffs = sp.symbols("M1:%d" % (k + 1))             # M1 .. Mk
    series = 1 + sum(coeffs[i] * z ** (i + 1) for i in range(k))
    poly = sp.expand(sp.numer(sp.cancel(sp.together(lmuz.subs(mu, series)))))
    sol, moments = {}, []
    for j in range(1, k + 1):
        cj = sp.expand(poly.coeff(z, j).subs(sol))
        unknown = [sym for sym in cj.free_symbols if sym in coeffs]
        if unknown:
            sol[unknown[0]] = sp.solve(cj, unknown[0])[0]
            moments.append(sp.simplify(sol[unknown[0]]))
        else:
            moments.append(sp.Integer(0))
    return moments


def _moments_fast(L, k):
    """MomF: D-finite linear recurrence for the moment sequence (RMTool fast path).

    Solve L_muz for the branch mu(z) with mu(0) = 1, build its holonomic
    representation, read the recurrence + initial terms, and unroll. Falls back to
    the series method if the holonomic machinery cannot handle this polynomial.
    """
    try:
        from sympy.holonomic import expr_to_holonomic
        lmuz = enc.to_encoding(L, "muz").expr
        branch = None
        for root in sp.solve(lmuz, mu):
            if sp.limit(root, z, 0) == 1:
                branch = root
                break
        if branch is None:
            raise ValueError("no mu(0)=1 branch")
        hol = expr_to_holonomic(branch, z, x0=0)
        ser = hol.series(n=k + 1)                       # uses the recurrence internally
        poly = ser.removeO() if hasattr(ser, "removeO") else ser
        return [sp.nsimplify(poly.coeff(z, j)) for j in range(1, k + 1)]
    except Exception:
        return _moments_series(L, k)


def moments(L, k, method="series"):
    """First ``k`` moments [M_1, ..., M_k] of the measure encoded by ``L`` (M_0 = 1).

    method="series" (MomS, exact order-by-order) or "fast" (MomF, D-finite recurrence).
    """
    if method == "series":
        return _moments_series(L, k)
    if method == "fast":
        return _moments_fast(L, k)
    raise ValueError("method must be 'series' or 'fast'")
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_extract.py -v`
Expected: 4 passed. (If `test_moments_fast_*` fails because SymPy's holonomic path
mis-handles a branch, the fallback should keep them passing; investigate any genuine
disagreement against the closed-form Catalan/Narayana values — do not weaken the test.)

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/extract.py tests/test_core/test_extract.py
git commit -m "feat(core): moment extraction (series MomS + holonomic MomF), Catalan/Narayana-checked"
```

---

## Task 7: `extract.py` — density (`Lmz2pdf`)

**Files:**
- Modify: `rmtool_py/core/extract.py`
- Test: `tests/test_core/test_extract.py` (append)

For each grid point `x`, evaluate `L_mz(·, x + iε)` as a univariate polynomial in `m`,
take all roots, and select the "right root" (physical Stieltjes branch: `Im m > 0`,
largest imaginary part). Density `ρ(x) = (1/π)·Im m`. All roots are always returned so
the choice is transparent; a `root_selector` hook overrides the default.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_core/test_extract.py`:

```python
import numpy as np


def test_density_wigner_peak_at_zero():
    # Semicircle rho(0) = sqrt(4)/ (2 pi) = 1/pi.
    info = extract.density(wigner_lmz(), [0.0])
    assert abs(info.density[0] - 1.0 / np.pi) < 1e-3


def test_density_wigner_zero_outside_support():
    info = extract.density(wigner_lmz(), [3.0])     # support is [-2, 2]
    assert abs(info.density[0]) < 1e-6


def test_density_mp_support_within_edges():
    # MP(1/4): support [(1-1/2)^2, (1+1/2)^2] = [0.25, 2.25].
    grid = list(np.arange(-0.5, 3.0, 0.02))
    info = extract.density(marchenko_pastur(sp.Rational(1, 4)), grid)
    positive = [x for x, d in zip(info.range, info.density) if d > 1e-4]
    assert min(positive) > 0.24 - 0.05
    assert max(positive) < 2.25 + 0.05


def test_density_returns_all_roots():
    info = extract.density(wigner_lmz(), [0.0])
    assert info.all_roots is not None and len(info.all_roots[0]) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_extract.py -k density -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'density'`.

- [ ] **Step 3: Implement `density`**

Append to `rmtool_py/core/extract.py` (add `import numpy as np` and
`from dataclasses import dataclass` near the top of the file):

```python
@dataclass
class PdfInfo:
    """Result of density extraction (RMTool ``pdfinfo``)."""
    range: list        # grid x-values
    density: list      # (1/pi) Im m at the selected right root
    all_roots: list    # list of all m-roots per grid point (transparency)


def _default_right_root(roots):
    """Physical Stieltjes branch: among Im>0 roots, the largest imaginary part."""
    upper = [r for r in roots if r.imag > 1e-9]
    if not upper:
        return 0.0 + 0.0j
    return max(upper, key=lambda r: r.imag)


def density(L, grid, eps=1e-8, root_selector=None):
    """Limiting eigenvalue density on ``grid`` (RMTool ``Lmz2pdf``).

    For each x, solve L_mz(m, x + i eps) for all roots in m and report
    rho(x) = (1/pi) Im m_right.  All roots are returned in ``all_roots``.
    ``root_selector(roots) -> complex`` overrides the default branch choice.
    """
    select = root_selector or _default_right_root
    mvar = L.var1
    poly_m = sp.Poly(L.expr, mvar)
    zvar = L.var2
    rng, dens, allroots = [], [], []
    for x in grid:
        zc = complex(x) + 1j * eps
        coeffs = [complex(c.subs(zvar, zc)) for c in poly_m.all_coeffs()]
        roots = list(np.roots(coeffs)) if len(coeffs) > 1 else []
        rng.append(float(x))
        allroots.append(roots)
        dens.append(max(select(roots).imag / np.pi, 0.0))
    return PdfInfo(range=rng, density=dens, all_roots=allroots)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_extract.py -v`
Expected: 8 passed (4 moments + 4 density).

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/extract.py tests/test_core/test_extract.py
git commit -m "feat(core): density extraction (Lmz2pdf) with right-root selection + pdfinfo"
```

---

## Task 8: `measure.py` — the `AlgebraicMeasure` OO facade

**Files:**
- Create: `rmtool_py/core/measure.py`
- Test: `tests/test_core/test_measure.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core/test_measure.py`:

```python
import sympy as sp

from rmtool_py.core.measure import AlgebraicMeasure as AM

m, z, c = sp.symbols("m z c")


def test_constructors_and_eq_proportional():
    a = AM.wigner()
    # 2*(m^2 + m z + 1) is the same measure (proportional L_mz).
    from rmtool_py.core.polynomial import BivariatePolynomial
    b = AM(BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z))
    assert a == b
    assert hash(a) == hash(b)


def test_distinct_measures_not_equal():
    assert AM.wigner() != AM.marchenko_pastur(sp.Rational(1, 2))


def test_operator_add_is_free_sum_eq_1_2():
    out = AM.wigner() + AM.marchenko_pastur(sp.Rational(1, 2))
    from rmtool_py.core.polynomial import BivariatePolynomial
    expected = BivariatePolynomial(
        m ** 3 + (z + 2) * m ** 2 + (2 * z - 1) * m + 2, m, z)
    assert out.lmz.is_proportional_to(expected)


def test_operator_mul_is_free_product_eq_1_4():
    out = AM.wigner() * AM.marchenko_pastur(sp.Rational(1, 2))
    from rmtool_py.core.polynomial import BivariatePolynomial
    expected = BivariatePolynomial(
        m ** 4 * z ** 2 - 2 * m ** 3 * z + m ** 2 + 4 * m * z + 4, m, z)
    assert out.lmz.is_proportional_to(expected)


def test_invert_and_scale_shift():
    a = AM.atomic([1], [3])
    assert (~a) == AM.atomic([1], [sp.Rational(1, 3)])
    assert a.scale(2) == AM.atomic([1], [6])
    assert a.shift(2) == AM.atomic([1], [5])
    assert a.square() == AM.atomic([1], [9])


def test_times_wishart_method():
    out = AM.wigner().times_wishart(sp.Rational(1, 2))
    expected = AM.wigner() * AM.marchenko_pastur(sp.Rational(1, 2))
    assert out == expected


def test_moments_and_density_and_latex():
    mp = AM.marchenko_pastur(sp.Rational(1, 2))
    assert [sp.nsimplify(x) for x in mp.moments(3)] == [1, sp.Rational(3, 2), sp.Rational(11, 4)]
    info = mp.density([1.0])
    assert info.density[0] >= 0.0
    assert isinstance(mp.latex(), str) and "m" in mp.latex()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_core/test_measure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.core.measure'`.

- [ ] **Step 3: Implement `AlgebraicMeasure`**

Create `rmtool_py/core/measure.py`:

```python
"""AlgebraicMeasure: the object-oriented facade over the polynomial engine."""

import sympy as sp

from .polynomial import BivariatePolynomial
from . import atoms, operations, extract

m, z = sp.symbols("m z")


class AlgebraicMeasure:
    """A limiting eigenvalue distribution, carried as its canonical L_mz polynomial.

    Operators: a + b (free additive), a * b (free multiplicative), ~a (inverse).
    Equality is proportional (two proportional L_mz are the same measure) and hashable.
    """

    def __init__(self, lmz):
        if not isinstance(lmz, BivariatePolynomial):
            raise TypeError("AlgebraicMeasure expects a BivariatePolynomial in (m, z)")
        self.lmz = lmz.normalize()

    # --- constructors -----------------------------------------------------
    @classmethod
    def wigner(cls):
        return cls(atoms.wigner_lmz())

    @classmethod
    def marchenko_pastur(cls, c):
        return cls(atoms.marchenko_pastur(c))

    @classmethod
    def atomic(cls, weights, points):
        return cls(atoms.atomic_lmz(weights, points))

    # --- operators --------------------------------------------------------
    def __add__(self, other):
        return AlgebraicMeasure(operations.add(self.lmz, other.lmz))

    def __mul__(self, other):
        return AlgebraicMeasure(operations.mult(self.lmz, other.lmz))

    def __invert__(self):
        return AlgebraicMeasure(operations.inverse(self.lmz))

    # --- deterministic methods -------------------------------------------
    def scale(self, alpha):
        return AlgebraicMeasure(operations.scale(self.lmz, alpha))

    def shift(self, alpha):
        return AlgebraicMeasure(operations.shift(self.lmz, alpha))

    def mobius(self, p, q, r, s):
        return AlgebraicMeasure(operations.mobius(self.lmz, p, q, r, s))

    def square(self):
        return AlgebraicMeasure(operations.square(self.lmz))

    def transpose(self, c):
        return AlgebraicMeasure(operations.transpose(self.lmz, c))

    # --- stochastic methods ----------------------------------------------
    def times_wishart(self, c):
        return AlgebraicMeasure(operations.times_wishart(self.lmz, c))

    def gram_wishart(self, c, s):
        return AlgebraicMeasure(operations.gram_wishart(self.lmz, c, s))

    def compress(self, c):
        return AlgebraicMeasure(operations.compress(self.lmz, c))

    # --- extraction -------------------------------------------------------
    def density(self, grid, **kw):
        return extract.density(self.lmz, grid, **kw)

    def moments(self, k, method="series"):
        return extract.moments(self.lmz, k, method=method)

    # --- presentation -----------------------------------------------------
    def latex(self):
        return sp.latex(self.lmz.expr)

    def pretty(self):
        return sp.pretty(self.lmz.expr)

    def coefficient_table(self):
        """Matrix of coefficients of L_mz: entry [i, j] = coeff of m^i z^j (RMTool TLmz)."""
        poly = sp.Poly(self.lmz.expr, self.lmz.var1, self.lmz.var2)
        di = poly.degree(self.lmz.var1)
        dj = poly.degree(self.lmz.var2)
        return sp.Matrix(di + 1, dj + 1,
                         lambda i, j: poly.coeff_monomial(self.lmz.var1 ** i * self.lmz.var2 ** j))

    # --- equality (proportional, hashable) -------------------------------
    def _key(self):
        return sp.srepr(self.lmz.canonical_form())

    def __eq__(self, other):
        return isinstance(other, AlgebraicMeasure) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return f"AlgebraicMeasure({self.lmz.expr})"
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_core/test_measure.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/measure.py tests/test_core/test_measure.py
git commit -m "feat(core): AlgebraicMeasure OO facade with proportional, hashable equality"
```

---

## Task 9: `compat.py` — RMTool-exact names + §1.3/§2.5 acceptance

**Files:**
- Create: `rmtool_py/compat.py`
- Test: `tests/test_compat.py` (create)

RMTool functions operate on the bivariate polynomial directly (a `BivariatePolynomial`
here), matching the Users Guide §2.2 table verbatim.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compat.py`:

```python
import sympy as sp
import numpy as np

from rmtool_py import compat as rt

m, z, c = sp.symbols("m z c")


def test_wishartpol_and_wignerpol():
    assert rt.wishartpol(sp.Rational(1, 2)).is_proportional_to(rt.wishartpol(sp.Rational(1, 2)))
    assert sp.Poly(rt.wignerpol().expr, m).degree() == 2


def test_quickstart_wishart_moments():
    # Sec 1.3: b = wishartpol(0.5); Lmz2MomS(b, 10).  MP(1/2): M1..M3 = 1, 1.5, 2.75.
    moms = rt.Lmz2MomS(rt.wishartpol(sp.Rational(1, 2)), 10)
    assert len(moms) == 10
    assert [sp.nsimplify(x) for x in moms[:3]] == [1, sp.Rational(3, 2), sp.Rational(11, 4)]


def test_quickstart_wigner_moments():
    moms = rt.Lmz2MomS(rt.wignerpol(), 10)
    assert [sp.nsimplify(x) for x in moms[:5]] == [0, 1, 0, 2, 0]


def test_quickstart_density_runs():
    info = rt.Lmz2pdf(rt.wishartpol(sp.Rational(1, 2)), list(np.arange(-0.05, 5.0, 0.05)))
    assert hasattr(info, "range") and hasattr(info, "density")
    assert len(info.range) == len(info.density)


def test_quickstart_aplusb_symbolic_c():
    # Sec 1.3: b = AplusB(wignerpol, wishartpol(c)); degree 3 in m.
    b = rt.AplusB(rt.wignerpol(), rt.wishartpol(c))
    assert sp.Poly(b.expr, m).degree() == 3


def test_section_2_5_atimesb_runs():
    # Sec 2.5: b3 = AtimesB(wishartpol(c), wignerpol); then pretty/latex/TLmz.
    b3 = rt.AtimesB(rt.wishartpol(c), rt.wignerpol())
    assert isinstance(rt.latex(b3), str)
    assert isinstance(rt.pretty(b3), str)
    assert rt.TLmz(b3).shape[0] >= 1


def test_deterministic_compat_names():
    L = rt.wignerpol()
    # compat returns BivariatePolynomial (no __eq__); compare with is_proportional_to.
    assert rt.shiftA(rt.wignerpol(), 1).is_proportional_to(rt.shiftA(rt.wignerpol(), 1))
    for fn in (rt.invA, rt.squareA):
        assert isinstance(fn(L), type(L))
    assert isinstance(rt.scaleA(L, 2), type(L))
    assert isinstance(rt.mobiusA(L, 1, 0, 0, 1), type(L))
    assert isinstance(rt.transposeA(L, sp.Rational(1, 2)), type(L))
    assert isinstance(rt.AtimesWish(L, sp.Rational(1, 2)), type(L))
    assert isinstance(rt.AgramWish(L, c, sp.Symbol("s")), type(L))
    assert isinstance(rt.compressA(rt.wishartpol(sp.Rational(1, 4)), sp.Rational(1, 2)), type(L))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_compat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.compat'`.

- [ ] **Step 3: Implement `compat.py`**

Create `rmtool_py/compat.py`:

```python
"""RMTool-exact function names (Users Guide v1.0 Sec 2.2) over the Phase-2 core.

Each function takes/returns a BivariatePolynomial in (m, z), matching RMTool's
convention that the user manipulates the bivariate polynomial directly. ``__eq__``
of the equality helpers comes from AlgebraicMeasure; raw polynomials compare with
``.is_proportional_to``. These are documented aliases, not new behaviour.
"""

import sympy as sp

from .core import atoms, operations, extract
from .core.measure import AlgebraicMeasure


# --- measure constructors -------------------------------------------------
def wignerpol():
    """Semicircle / Wigner L_mz (RMTool ``wignerpol``)."""
    return atoms.wigner_lmz()


def wishartpol(c):
    """Marcenko-Pastur / Wishart L_mz with ratio ``c`` (RMTool ``wishartpol``)."""
    return atoms.marchenko_pastur(c)


# --- deterministic laws ---------------------------------------------------
def invA(L):
    return operations.inverse(L)


def shiftA(L, alpha):
    return operations.shift(L, alpha)


def scaleA(L, alpha):
    return operations.scale(L, alpha)


def mobiusA(L, p, q, r, s):
    return operations.mobius(L, p, q, r, s)


def transposeA(L, c):
    return operations.transpose(L, c)


def squareA(L):
    return operations.square(L)


# --- stochastic laws ------------------------------------------------------
def AtimesWish(L, c):
    return operations.times_wishart(L, c)


def AgramWish(L, c, s):
    return operations.gram_wishart(L, c, s)


def AplusB(La, Lb):
    return operations.add(La, Lb)


def AtimesB(La, Lb):
    return operations.mult(La, Lb)


def compressA(La, c):
    return operations.compress(La, c)


# --- extraction -----------------------------------------------------------
def Lmz2pdf(L, xx):
    return extract.density(L, xx)


def Lmz2MomS(L, n):
    return extract.moments(L, n, method="series")


def Lmz2MomF(L, n):
    return extract.moments(L, n, method="fast")


# --- presentation ---------------------------------------------------------
def pretty(L):
    return sp.pretty(L.expr)


def latex(L):
    return sp.latex(L.expr)


def TLmz(L):
    """Matrix of coefficients of L_mz (RMTool ``TLmz``)."""
    return AlgebraicMeasure(L).coefficient_table()
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/test_compat.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/compat.py tests/test_compat.py
git commit -m "feat: RMTool-exact compat shim; user-guide Sec 1.3/2.5 acceptance tests"
```

---

## Task 10: Wire up exports and run the full suite

**Files:**
- Modify: `rmtool_py/core/__init__.py`
- Modify: `rmtool_py/__init__.py`

- [ ] **Step 1: Update `rmtool_py/core/__init__.py`**

Replace its contents:

```python
"""Core polynomial-method engine."""

from .polynomial import BivariatePolynomial
from . import encodings, algebra, atoms, operations, extract, measure
from .atoms import wigner_lmz, atomic_lmz, marchenko_pastur
from .measure import AlgebraicMeasure

__all__ = [
    "BivariatePolynomial", "AlgebraicMeasure",
    "encodings", "algebra", "atoms", "operations", "extract", "measure",
    "wigner_lmz", "atomic_lmz", "marchenko_pastur",
]
```

- [ ] **Step 2: Update `rmtool_py/__init__.py`**

Read the current file first; then ensure it exposes the facade and compat without
breaking existing `matrices`/`viz` imports. Append (or merge into the existing
`__all__`) these lines:

```python
from .core import AlgebraicMeasure
from . import compat

# extend (do not clobber) any existing __all__
try:
    __all__  # noqa: F821
except NameError:
    __all__ = []
__all__ += ["AlgebraicMeasure", "compat"]
```

- [ ] **Step 3: Verify imports**

Run:
```bash
python3 -c "from rmtool_py import AlgebraicMeasure, compat; from rmtool_py.core import operations, extract, measure, marchenko_pastur; print('imports ok')"
```
Expected: `imports ok` with no error.

- [ ] **Step 4: Run the entire suite**

Run: `python3 -m pytest tests -q`
Expected: all green — 28 Phase-1 + 5 encodings + 2 atoms + 4 polynomial + 13 operations
+ 8 extract + 7 measure + 7 compat ≈ 74 passed. (Exact count may vary by ±a few; zero
failures is the bar.)

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/__init__.py rmtool_py/core/__init__.py
git commit -m "feat: export AlgebraicMeasure + compat from package root"
```

---

## Phase 2 completion criteria

- [ ] `python3 -m pytest tests -q` → all green, no skips of the ground-truth tests.
- [ ] `python3 -c "from rmtool_py import AlgebraicMeasure, compat"` → no error.
- [ ] muz/etaz reproduce Table 2(a/b/c) (etaz-MP uses the corrected `+z` form) and round-trip.
- [ ] `marchenko_pastur(c)` edges = `(1±√c)²`.
- [ ] `add`→Eq. 1.2 exactly; `mult`/`times_wishart`→Eq. 1.4 (∝); `compress`→Eq. 10.9 exactly; gram_wishart(δ₀,c,s)∝MP(c).scale(s); deterministic laws verified on atom points.
- [ ] moments: series & fast agree and match Catalan (Wigner) / Narayana (MP).
- [ ] density: Wigner `ρ(0)≈1/π`, zero outside `[−2,2]`; MP support within edges; all roots returned.
- [ ] `AlgebraicMeasure` equality proportional + hashable; operators `+ * ~` and all methods work.
- [ ] compat §1.3 sequence (`wishartpol`, `Lmz2MomS`, `Lmz2pdf`, `wignerpol`, `AplusB`) and §2.5 (`AtimesB`, `pretty`, `latex`, `TLmz`) run and reproduce documented behaviour.

---

## Roadmap — Phase 3 (authored when reached)

`finance/` (correlation_matrix, remove_market_mode, empirical_density, fit_marchenko_pastur,
information_eigenvalues, component_distribution, porter_thomas_pdf, inverse_participation_ratio,
factor_model_returns), `viz/figures.py` (Laloux Figs 1 & 2), cleaned seedable `matrices/`.
Gold-standard test: simulate a factor model with known population Σ; assert empirical sample
spectrum ≈ `atomic(Σ-spectrum).times_wishart(N/T)` within finite-N tolerance.
```
