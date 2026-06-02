# RMTool-Py Phase 1: Core Engine Foundations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the symbolic substrate of the polynomial method — a normalized bivariate-polynomial type, the six transform encodings with conversions between them, and the `⊞`/`⊠` binary operators on algebraic functions — all verified against ground-truth values from the paper.

**Architecture:** A `BivariatePolynomial` wrapper around SymPy carries the irreducibility normalization (`irreducLuv`: clear denominators + square-free). `encodings.py` implements the six representations (Stieltjes/Cauchy/R/S/moment/eta) and the paper's Table-3 conversions as variable substitutions + normalization. `algebra.py` implements companion-matrix construction and the `⊞`/`⊠` operators (Prop. 4.6 / Table 5). `atoms.py` provides the Wigner and atomic-measure constructors (the only two whose exact polynomials we can verify without density extraction; Marčenko–Pastur arrives in Phase 2 where edge tests validate it).

**Tech Stack:** Python 3.8, SymPy 1.13, pytest. (NumPy/SciPy/Matplotlib already present; used in later phases.)

**Reference:** Rao & Edelman, *The Polynomial Method for Random Matrices*, Found. Comput. Math. 8 (2008) 649–702 — Tables 1–6, Eqs. 3.3–3.11, Fig. 3. Spec: `docs/superpowers/specs/2026-06-01-rmtool-py-polynomial-engine-design.md`.

**Ground-truth anchors used as tests (all read directly from the paper):**
- Wigner / semicircle across all six encodings (Table 2(c)):
  `L_mz = m² + mz + 1`, `L_gz = g² − gz + 1`, `L_rg = r − g`, `L_sy = s²y − 1`, `L_μz = μ²z² − μ + 1`, `L_ηz = z²η² − η + 1`.
- Atomic example (Eq. 3.7): `F = 0.5·𝟙_{[0,∞)} + 0.5·𝟙_{[1,∞)}` (atoms at `0` and `1`, weights `½,½`) ⇒ `L_mz = m(2z² − 2z) − (1 − 2z)`.
- `⊞`/`⊠` worked example (Table 6) — exact coefficient tables, transcribed in Task 5.

---

## File structure (created in this plan)

```
rmtool_py/
  core/
    __init__.py        # exports BivariatePolynomial, encodings, algebra, atoms
    polynomial.py      # BivariatePolynomial + normalization + proportionality equality
    encodings.py       # mz/gz/rg/sy encodings + Table-3 conversions + to_encoding() (muz/etaz: Plan 2)
    algebra.py         # companion_matrix, kron, boxplus (⊞), boxtimes (⊠)
    atoms.py           # wigner_lmz(), atomic_lmz(weights, points)
tests/
  test_core/
    __init__.py
    test_polynomial.py
    test_atoms.py
    test_encodings.py
    test_algebra.py
pyproject.toml         # MODIFY: add sympy runtime dep + [project.optional-dependencies] dev = pytest + pytest config
```

Note: `rmtool_py/matrices/` and the old `tests/test_matrices/test_wigner.py` demo script are untouched in Phase 1 (cleaned in Phase 3).

---

## Task 1: Project setup — dependencies, core package, pytest

**Files:**
- Modify: `pyproject.toml`
- Create: `rmtool_py/core/__init__.py`
- Create: `tests/test_core/__init__.py`

- [ ] **Step 1: Add SymPy + dev deps and pytest config to `pyproject.toml`**

Add `"sympy>=1.11"` to `dependencies`, and append these two sections at the end of the file:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

The `dependencies` list becomes:

```toml
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "sympy>=1.11",
]
```

- [ ] **Step 2: Create the core package init**

Create `rmtool_py/core/__init__.py`:

```python
"""Core polynomial-method engine (bivariate-polynomial substrate)."""

from .polynomial import BivariatePolynomial
from . import encodings, algebra, atoms

__all__ = ["BivariatePolynomial", "encodings", "algebra", "atoms"]
```

This import will fail until later tasks create the modules; that is expected. Create `tests/test_core/__init__.py` as an empty file.

- [ ] **Step 3: Verify the toolchain**

Run: `python3 -c "import sympy, pytest; print(sympy.__version__)"`
Expected: prints a SymPy version `>= 1.11` with no ImportError. (If `pytest` is missing: `python3 -m pip install pytest`.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml rmtool_py/core/__init__.py tests/test_core/__init__.py
git commit -m "chore: add sympy/pytest deps and core package skeleton"
```

---

## Task 2: `BivariatePolynomial` — normalization + proportional equality

**Files:**
- Create: `rmtool_py/core/polynomial.py`
- Test: `tests/test_core/test_polynomial.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core/test_polynomial.py`:

```python
import sympy as sp
from rmtool_py.core.polynomial import BivariatePolynomial

m, z = sp.symbols("m z")


def test_clear_denominators_keeps_numerator():
    # m - 0.5/(1-z) - 0.5/(2-z)  ->  numerator after clearing
    bp = BivariatePolynomial(m - sp.Rational(1, 2) / (1 - z) - sp.Rational(1, 2) / (2 - z), m, z)
    cleared = bp.clear_denominators()
    # No denominators left in m, z
    assert sp.denom(sp.together(cleared.expr)) == 1


def test_make_squarefree_removes_repeated_factor():
    bp = BivariatePolynomial((m - z) ** 2 * (m + 1), m, z)
    sf = bp.make_squarefree()
    # Square-free part wrt m: (m - z)(m + 1), proportional to that
    expected = (m - z) * (m + 1)
    assert sf.is_proportional_to(BivariatePolynomial(expected, m, z))


def test_normalize_is_idempotent():
    bp = BivariatePolynomial((m - z) ** 2 / (1 - z), m, z)
    once = bp.normalize()
    twice = once.normalize()
    assert once.is_proportional_to(twice)


def test_proportional_equality():
    a = BivariatePolynomial(2 * m ** 2 + 2 * m * z + 2, m, z)
    b = BivariatePolynomial(m ** 2 + m * z + 1, m, z)
    assert a.is_proportional_to(b)
    c = BivariatePolynomial(m ** 2 + m * z + 2, m, z)
    assert not a.is_proportional_to(c)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_core/test_polynomial.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.core.polynomial'`.

- [ ] **Step 3: Implement `BivariatePolynomial`**

Create `rmtool_py/core/polynomial.py`:

```python
"""Bivariate-polynomial type carrying the irreducLuv normalization."""

import sympy as sp


class BivariatePolynomial:
    """A bivariate polynomial L(var1, var2) representing a measure encoding.

    ``var1`` is the *transform* variable (m, g, r, s, mu, eta) — the one we
    eliminate / take square-free part with respect to. ``var2`` is the argument
    variable (z, g, y).
    """

    def __init__(self, expr, var1, var2):
        self.var1 = sp.sympify(var1)
        self.var2 = sp.sympify(var2)
        self.expr = sp.expand(sp.sympify(expr))

    @property
    def vars(self):
        return (self.var1, self.var2)

    def clear_denominators(self):
        """Multiply out denominators, keep the numerator (RMTool ``numden``).

        ``sp.cancel`` first reduces the rational expression to lowest terms so a
        factor shared by numerator and denominator (e.g. an ``(m*z + 1)`` left
        over from a transform substitution) is removed rather than retained.
        """
        num, _den = sp.fraction(sp.cancel(sp.together(self.expr)))
        return BivariatePolynomial(sp.expand(num), self.var1, self.var2)

    def make_squarefree(self, var=None):
        """Divide out repeated/content factors w.r.t. ``var`` (default var1).

        Implements the paper's irreducLuv square-free step exactly:
        ``L / gcd(L, dL/dvar)`` using SymPy's *multivariate* gcd, which also
        strips any spurious factor that does not involve ``var``.
        """
        var = self.var1 if var is None else sp.sympify(var)
        e = sp.expand(self.expr)
        common = sp.gcd(e, sp.diff(e, var))
        quotient = sp.cancel(e / common)
        return BivariatePolynomial(sp.expand(quotient), self.var1, self.var2)

    def normalize(self):
        """Full irreducLuv: clear denominators, then square-free w.r.t. both vars."""
        bp = self.clear_denominators()
        bp = bp.make_squarefree(self.var1)
        bp = bp.make_squarefree(self.var2)
        return bp

    def is_proportional_to(self, other):
        """True iff self.expr == c * other.expr for some nonzero constant c."""
        e1, e2 = sp.expand(self.expr), sp.expand(other.expr)
        if e2 == 0:
            return e1 == 0
        ratio = sp.cancel(e1 / e2)
        return len(ratio.free_symbols) == 0 and ratio != 0

    def __repr__(self):
        return f"BivariatePolynomial({self.expr}, {self.var1}, {self.var2})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_core/test_polynomial.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/polynomial.py tests/test_core/test_polynomial.py
git commit -m "feat(core): BivariatePolynomial with irreducLuv normalization"
```

---

## Task 3: `atoms.py` — Wigner and atomic-measure constructors

**Files:**
- Create: `rmtool_py/core/atoms.py`
- Test: `tests/test_core/test_atoms.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core/test_atoms.py`:

```python
import sympy as sp
from rmtool_py.core.atoms import wigner_lmz, atomic_lmz
from rmtool_py.core.polynomial import BivariatePolynomial

m, z = sp.symbols("m z")


def test_wigner_lmz():
    # Table 2(c): L_mz = m^2 + m z + 1
    assert wigner_lmz().is_proportional_to(BivariatePolynomial(m ** 2 + m * z + 1, m, z))


def test_atomic_two_point_eq_3_7():
    # Eq. 3.7: atoms at 0 and 1, weights 1/2, 1/2  ->  L_mz = m(2z^2 - 2z) - (1 - 2z)
    bp = atomic_lmz([sp.Rational(1, 2), sp.Rational(1, 2)], [0, 1])
    expected = BivariatePolynomial(m * (2 * z ** 2 - 2 * z) - (1 - 2 * z), m, z)
    assert bp.is_proportional_to(expected)


def test_atomic_single_point():
    # Single atom at a: m(z) = 1/(a - z)  ->  L_mz = m(a - z) - 1
    a = 3
    bp = atomic_lmz([1], [a])
    expected = BivariatePolynomial(m * (a - z) - 1, m, z)
    assert bp.is_proportional_to(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_core/test_atoms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.core.atoms'`.

- [ ] **Step 3: Implement `atoms.py`**

Create `rmtool_py/core/atoms.py`:

```python
"""Constructors for atomic and Wigner measure encodings (L_mz form)."""

import sympy as sp

from .polynomial import BivariatePolynomial

m, z = sp.symbols("m z")


def wigner_lmz():
    """Semicircle (Wigner) law. Table 2(c): L_mz = m^2 + m z + 1."""
    return BivariatePolynomial(m ** 2 + m * z + 1, m, z)


def atomic_lmz(weights, points):
    """Atomic measure sum_i w_i * delta(x - p_i).

    Stieltjes transform m(z) = sum_i w_i / (p_i - z); the L_mz polynomial is the
    numerator of (m - m(z)) after clearing denominators.
    """
    if len(weights) != len(points):
        raise ValueError("weights and points must have equal length")
    stieltjes = sum(sp.Rational(0) + w / (p - z) for w, p in zip(weights, points))
    bp = BivariatePolynomial(m - stieltjes, m, z)
    return bp.normalize()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_core/test_atoms.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/atoms.py tests/test_core/test_atoms.py
git commit -m "feat(core): wigner_lmz and atomic_lmz constructors"
```

---

## Task 4: `encodings.py` — the six representations and Table-3 conversions

**Files:**
- Create: `rmtool_py/core/encodings.py`
- Test: `tests/test_core/test_encodings.py`

This is the spine. We implement the conversions exactly as the paper's Table 3 substitutions and verify each against the Wigner Table-2(c) values (every encoding has a known target) plus round-trip identity.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core/test_encodings.py`:

```python
import sympy as sp
from rmtool_py.core.atoms import wigner_lmz
from rmtool_py.core.polynomial import BivariatePolynomial
from rmtool_py.core import encodings as enc

m, z, g, r, s, y = sp.symbols("m z g r s y")


def _prop(bp, expr, v1, v2):
    return bp.is_proportional_to(BivariatePolynomial(expr, v1, v2))


def test_wigner_encodings_gz_rg_sy():
    # Table 2(c) values for the four Phase-1 encodings (muz/etaz are Plan 2).
    lmz = wigner_lmz()
    assert _prop(enc.to_encoding(lmz, "gz"), g ** 2 - g * z + 1, g, z)
    assert _prop(enc.to_encoding(lmz, "rg"), r - g, r, g)
    assert _prop(enc.to_encoding(lmz, "sy"), s ** 2 * y - 1, s, y)


def test_roundtrip_mz_gz_mz():
    lmz = wigner_lmz()
    back = enc.to_encoding(enc.to_encoding(lmz, "gz"), "mz")
    assert back.is_proportional_to(lmz)


def test_roundtrip_mz_rg_mz():
    lmz = wigner_lmz()
    back = enc.to_encoding(enc.to_encoding(lmz, "rg"), "mz")
    assert back.is_proportional_to(lmz)


def test_roundtrip_mz_sy_mz():
    lmz = wigner_lmz()
    back = enc.to_encoding(enc.to_encoding(lmz, "sy"), "mz")
    assert back.is_proportional_to(lmz)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_core/test_encodings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.core.encodings'`.

- [ ] **Step 3: Implement `encodings.py`**

Create `rmtool_py/core/encodings.py`. The substitutions are transcribed from Table 3 (Eqs. 3.3–3.5 and the Maple `subs` code in the table). `var1` of each encoding is the transform variable; conversions are dict (simultaneous) substitutions followed by `normalize`.

```python
"""The six bivariate-polynomial encodings and the Table-3 conversions.

Encodings (paper Fig. 3 / Table 2):
    mz   Stieltjes m(z)       vars (m, z)
    gz   Cauchy    g(z)=-m    vars (g, z)
    rg   R-transform r(g)     vars (r, g)
    sy   S-transform s(y)     vars (s, y)
    muz  moment    mu(z)      vars (mu, z)
    etaz eta       eta(z)     vars (eta, z)
"""

import sympy as sp

from .polynomial import BivariatePolynomial

m, z, g, r, s, y = sp.symbols("m z g r s y")

# encoding name -> (transform var, argument var)
# Phase 1 implements the four encodings the substrate + free-additive convolution
# need. The moment (muz) and eta (etaz) encodings are added in Plan 2, where
# Table-3 rows V-VI are transcribed and pinned with round-trip + Table-2 tests.
ENCODING_VARS = {
    "mz": (m, z),
    "gz": (g, z),
    "rg": (r, g),
    "sy": (s, y),
}


def _wrap(expr, name):
    v1, v2 = ENCODING_VARS[name]
    return BivariatePolynomial(expr, v1, v2).normalize()


# --- direct conversions (Table 3) -----------------------------------------
# Each takes the source expr and returns the target expr (un-normalized);
# _wrap normalizes. Substitutions are simultaneous (SymPy dict subs).

def mz_to_gz(e):                 # I:  L_gz(g,z) = L_mz(-g, z)
    return e.subs({m: -g}, simultaneous=True)


def gz_to_mz(e):                 # I:  L_mz(m,z) = L_gz(-m, z)
    return e.subs({g: -m}, simultaneous=True)


def gz_to_rg(e):                 # II: L_rg(r,g) = L_gz(g, r + 1/g)
    return e.subs({z: r + 1 / g}, simultaneous=True)


def rg_to_gz(e):                 # II: L_gz(g,z) = L_rg(z - 1/g, g)
    return e.subs({r: z - 1 / g}, simultaneous=True)


def mz_to_sy(e):                 # IV (Eq. 3.5): L_sy = L_mz(-y*s, (y+1)/(s*y))
    return e.subs({m: -y * s, z: (y + 1) / (s * y)}, simultaneous=True)


def sy_to_mz(e):                 # IV: L_mz = L_sy(s=m/(z*m+1), y=-z*m-1)
    return e.subs({s: m / (z * m + 1), y: -z * m - 1}, simultaneous=True)


# --- conversion graph -------------------------------------------------------
# edge: (from, to) -> function
_EDGES = {
    ("mz", "gz"): mz_to_gz, ("gz", "mz"): gz_to_mz,
    ("gz", "rg"): gz_to_rg, ("rg", "gz"): rg_to_gz,
    ("mz", "sy"): mz_to_sy, ("sy", "mz"): sy_to_mz,
}

# adjacency for BFS shortest path
_ADJ = {}
for (a, b) in _EDGES:
    _ADJ.setdefault(a, []).append(b)


def _path(src, dst):
    from collections import deque
    if src == dst:
        return [src]
    seen, queue = {src}, deque([[src]])
    while queue:
        path = queue.popleft()
        for nxt in _ADJ.get(path[-1], []):
            if nxt in seen:
                continue
            if nxt == dst:
                return path + [nxt]
            seen.add(nxt)
            queue.append(path + [nxt])
    raise ValueError(f"no conversion path {src} -> {dst}")


def to_encoding(bp, target):
    """Convert a BivariatePolynomial to the named encoding, normalized."""
    # identify source by its var pair
    src = None
    for name, (v1, v2) in ENCODING_VARS.items():
        if (bp.var1, bp.var2) == (v1, v2):
            src = name
            break
    if src is None:
        raise ValueError(f"unknown source encoding for vars {bp.vars}")
    if src == target:
        return bp.normalize()
    expr = bp.expr
    steps = _path(src, target)
    for a, b in zip(steps, steps[1:]):
        expr = sp.expand(sp.together(_EDGES[(a, b)](expr)))
        # normalize at each hop to keep degrees down and clear denominators
        v1, v2 = ENCODING_VARS[b]
        expr = BivariatePolynomial(expr, v1, v2).normalize().expr
    return _wrap(expr, target)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_core/test_encodings.py -v`
Expected: 4 passed.

If a single encoding assertion fails, the bug is in that one substitution — re-check the corresponding row of the paper's Table 3 (the sign/argument), fix only that function, and re-run. The round-trip tests guard the inverse directions.

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/encodings.py tests/test_core/test_encodings.py
git commit -m "feat(core): six transform encodings and Table-3 conversions"
```

---

## Task 5: `algebra.py` — companion matrices, `⊞` and `⊠`

**Files:**
- Create: `rmtool_py/core/algebra.py`
- Test: `tests/test_core/test_algebra.py`

We verify against Table 6 of the paper, which gives explicit `⊞`/`⊠` of two sample polynomials. We test by *evaluating* the resulting algebraic function: the roots of `L³(u, v₀)=0` at a fixed `v₀` must equal the sums (resp. products) of the roots of the two input polynomials at `v₀`. This validates `⊞`/`⊠` independently of how the paper happened to print the coefficient table.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core/test_algebra.py`:

```python
import sympy as sp
from rmtool_py.core.polynomial import BivariatePolynomial
from rmtool_py.core.algebra import companion_matrix, boxplus, boxtimes

u, v = sp.symbols("u v")

# Table 6 inputs:
#   L1(u,v) = u^2 + u(1 - v) + v^2
#   L2(u,v) = u^2 (v^2 - 3v + 1) + u(1 + v) + v^2
L1 = BivariatePolynomial(u ** 2 + u * (1 - v) + v ** 2, u, v)
L2 = BivariatePolynomial(u ** 2 * (v ** 2 - 3 * v + 1) + u * (1 + v) + v ** 2, u, v)


def _key(c):
    c = complex(c)
    return (round(c.real, 6), round(c.imag, 6))


def _roots_at(bp, v0):
    p = sp.Poly(bp.expr.subs(v, v0), u)
    return sorted((complex(c) for c in p.all_roots()), key=_key)


def _sorted_complex(vals):
    return sorted((complex(c) for c in vals), key=_key)


def test_companion_charpoly_recovers_polynomial():
    C = companion_matrix(L1, u)
    # char poly det(u I - C) is proportional to L1 (monic in u)
    n = C.shape[0]
    chi = (u * sp.eye(n) - C).det()
    assert BivariatePolynomial(chi, u, v).is_proportional_to(L1)


def test_boxplus_values_are_sums_of_roots():
    L3 = boxplus(L1, L2, u)
    v0 = 2  # generic point where all polys have distinct roots
    r1 = _roots_at(L1, v0)
    r2 = _roots_at(L2, v0)
    expected = _sorted_complex([a + b for a in r1 for b in r2])
    got = _roots_at(L3, v0)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert abs(a - b) < 1e-6


def test_boxtimes_values_are_products_of_roots():
    L3 = boxtimes(L1, L2, u)
    v0 = 2
    r1 = _roots_at(L1, v0)
    r2 = _roots_at(L2, v0)
    expected = _sorted_complex([a * b for a in r1 for b in r2])
    got = _roots_at(L3, v0)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert abs(a - b) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_core/test_algebra.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rmtool_py.core.algebra'`.

- [ ] **Step 3: Implement `algebra.py`**

Create `rmtool_py/core/algebra.py` (companion matrix from Table 4; `⊞`/`⊠` from Prop. 4.6 / Table 5):

```python
"""Binary operators on algebraic functions: boxplus and boxtimes.

Given two bivariate polynomials L1(u,v), L2(u,v) defining algebraic functions
u1(v), u2(v) (as roots in u), boxplus returns the polynomial whose roots are
u1(v)+u2(v) and boxtimes the one whose roots are u1(v)*u2(v) (Prop. 4.6).
"""

import sympy as sp

from .polynomial import BivariatePolynomial


def companion_matrix(bp, u):
    """Companion matrix C_u of bp treated as a polynomial in u (Table 4).

    Char poly det(u I - C_u) equals bp made monic in u.
    """
    u = sp.sympify(u)
    poly = sp.Poly(sp.expand(bp.expr), u)
    coeffs = poly.all_coeffs()           # leading first: [l_Du, ..., l_1, l_0]
    deg = poly.degree()
    lead = coeffs[0]
    C = sp.zeros(deg, deg)
    for i in range(1, deg):
        C[i, i - 1] = 1
    for j in range(deg):                 # coeff of u^j is coeffs[deg - j]
        l_j = coeffs[deg - j]
        C[j, deg - 1] = sp.cancel(-l_j / lead)
    return C


def _kron(A, B):
    ar, ac = A.shape
    br, bc = B.shape
    M = sp.zeros(ar * br, ac * bc)
    for i in range(ar):
        for j in range(ac):
            M[i * br:(i + 1) * br, j * bc:(j + 1) * bc] = A[i, j] * B
    return M


def _charpoly(C, u, other_var):
    n = C.shape[0]
    chi = sp.expand((u * sp.eye(n) - C).det())
    return BivariatePolynomial(chi, u, other_var).normalize()


def _other_var(bp, u):
    others = [s for s in (bp.var1, bp.var2) if s != sp.sympify(u)]
    return others[0]


def boxplus(bp1, bp2, u):
    """u1(v) + u2(v): eigenvalues of (C1 ⊗ I) + (I ⊗ C2) (Prop. 4.6.1)."""
    u = sp.sympify(u)
    other = _other_var(bp1, u)
    C1 = companion_matrix(bp1, u)
    C2 = companion_matrix(bp2, u)
    C3 = _kron(C1, sp.eye(C2.shape[0])) + _kron(sp.eye(C1.shape[0]), C2)
    return _charpoly(C3, u, other)


def boxtimes(bp1, bp2, u):
    """u1(v) * u2(v): eigenvalues of C1 ⊗ C2 (Prop. 4.6.2)."""
    u = sp.sympify(u)
    other = _other_var(bp1, u)
    C1 = companion_matrix(bp1, u)
    C2 = companion_matrix(bp2, u)
    C3 = _kron(C1, C2)
    return _charpoly(C3, u, other)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_core/test_algebra.py -v`
Expected: 3 passed. (The determinant of an up-to-6×6 symbolic matrix in one parameter is fast.)

- [ ] **Step 5: Commit**

```bash
git add rmtool_py/core/algebra.py tests/test_core/test_algebra.py
git commit -m "feat(core): companion matrices, boxplus and boxtimes operators"
```

---

## Task 6: Integration check — encodings + algebra compose

**Files:**
- Test: `tests/test_core/test_encodings.py` (append)

This proves the pieces fit: the R-transform of the semicircle is `r − g` (linear), and free additive convolution of two semicircles is done by `⊞` in the `rg` encoding. Doubling a semicircle via `⊞` of `L_rg` with itself, then converting back to `mz`, must give a valid scaled-semicircle polynomial of degree 2 in `m`.

- [ ] **Step 1: Append the failing test**

Append to `tests/test_core/test_encodings.py`:

```python
from rmtool_py.core.algebra import boxplus


def test_free_sum_of_two_semicircles_is_degree_two():
    # L_rg of Wigner is r - g (linear). boxplus over g of two copies, back to mz.
    lmz = wigner_lmz()
    lrg = enc.to_encoding(lmz, "rg")          # vars (r, g)
    summed_rg = boxplus(lrg, lrg, r)          # add R-transforms over g
    back = enc.to_encoding(
        BivariatePolynomial(summed_rg.expr, r, g), "mz"
    )
    # result is a genuine Stieltjes polynomial: degree 2 in m
    assert sp.Poly(back.expr, m).degree() == 2
```

- [ ] **Step 2: Run the test to verify it fails or errors**

Run: `python3 -m pytest tests/test_core/test_encodings.py::test_free_sum_of_two_semicircles_is_degree_two -v`
Expected: initially FAIL only if a prior task has a bug; if Tasks 4–5 are correct it may already pass. Either way, do not edit production code to force it — investigate any failure against the paper.

- [ ] **Step 3: (If needed) fix and re-run**

If it fails, the failure localizes to a conversion or to `⊞`; fix the offending function per the paper and re-run the full core suite: `python3 -m pytest tests/test_core -v` (Expected: all passed).

- [ ] **Step 4: Commit**

```bash
git add tests/test_core/test_encodings.py
git commit -m "test(core): integration check — free sum via R-transform + boxplus"
```

---

## Phase 1 completion criteria

- [ ] `python3 -m pytest tests/test_core -v` → all green.
- [ ] `python3 -c "from rmtool_py.core import BivariatePolynomial, encodings, algebra, atoms"` → no error.
- [ ] The four Phase-1 Wigner encodings (`mz`/`gz`/`rg`/`sy`) reproduce Table 2(c) and round-trip; the atomic example reproduces Eq. 3.7; `boxplus`/`boxtimes` reproduce root sums/products.

---

## Roadmap — subsequent plans (authored in detail when reached)

These are intentionally *not* yet broken into bite-sized steps: their exact code depends on the paper's Tables 7–9 (transcribed against the Phase-1 substrate) and on API decisions that firm up once Phase 1 runs. Each remains an independently shippable, testable unit.

**Plan 2 — Operations, extraction & OO facade (the full RMTool calculator).**
- `encodings` — add the moment (`muz`) and eta (`etaz`) encodings (Table-3 rows V-VI), pinned with round-trip + Table-2(c) value tests (`L_muz`, `L_etaz`). Deferred from Phase 1 because their exact substitution signs must be transcribed from the paper, not guessed.
- `atoms.marchenko_pastur(c)` — added here so its density-edge test validates the polynomial.
- `operations.py` — deterministic laws (`shift`, `scale`, `inverse`, `mobius`, `transpose`, `square`) as direct `L_mz` substitutions; stochastic laws (`add` via `rg`+`⊞`, `mult` via `sy`+`⊠`, `times_wishart`, `gram_wishart`, `compress`) transcribed from Tables 7–9. **Tests:** the paper's worked polynomials `L_mz^{A+B}` (Eq. 1.2) and `L_mz^{AB}` (Eq. 1.4).
- `extract.py` — `density` (root-finding on `L_mz(m, x+iε)`, right-root selection + `pdfinfo` struct) and `moments` (series + D-finite recursion, cross-checked). **Tests:** closed-form Wigner & MP density/moments; MomS vs MomF agreement.
- `measure.AlgebraicMeasure` — OO facade with operators (`+`, `*`, `~`, `.scale`, `.shift`, …) over the canonical `L_mz`; `.latex/.pretty/.coefficient_table`.
- `compat.py` — RMTool-exact names. **Acceptance test:** user-guide §1.3 quick-start reproduces documented output.

**Plan 3 — Finance / empirical layer + viz + factor-model validation (the practitioner deliverable).**
- `finance/correlation.py` (`correlation_matrix`, `remove_market_mode`), `finance/spectrum.py` (`empirical_density`, `mp_edges`, `fit_marchenko_pastur`, `information_eigenvalues`), `finance/eigenvectors.py` (`component_distribution`, `porter_thomas_pdf`, `inverse_participation_ratio`), `finance/data.py` (`factor_model_returns`).
- `viz/figures.py` (`plot_spectrum` → Fig. 1, `plot_eigenvector_distribution` → Fig. 2).
- Clean `matrices/random_matrices.py` (seedable RNG; resolve `sigma`); retire the demo script.
- **Gold-standard test:** simulate a factor model with known population Σ; assert the empirical sample spectrum matches the engine's `atomic(Σ-spectrum).times_wishart(N/T)` prediction within finite-N tolerance (plain MP, Σ=I, is the special case).
