"""The bivariate-polynomial encodings and the Table-3 conversions.

Encodings (paper Fig. 3 / Table 2):
    mz   Stieltjes m(z)       vars (m, z)
    gz   Cauchy    g(z)=-m    vars (g, z)
    rg   R-transform r(g)     vars (r, g)
    sy   S-transform s(y)     vars (s, y)

The moment (muz) and eta (etaz) encodings (Table-3 rows V-VI) are added in a
later plan, pinned with round-trip + Table-2 tests.
"""

import sympy as sp
from collections import deque

from .polynomial import BivariatePolynomial

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


# --- direct conversions (Table 3) -----------------------------------------
# Each takes the source expr and returns the target expr (un-normalized).
# Substitutions are simultaneous (SymPy dict subs).

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


def mz_to_muz(e):                # V:  Lmuz = Lmz(-mu*z, 1/z)
    return e.subs(z, 1 / z).subs(m, -mu * z)


def muz_to_mz(e):                # V:  Lmz = Lmuz(-m*z, 1/z)
    return e.subs(z, 1 / z).subs(mu, -m * z)


def mz_to_etaz(e):               # VI: Letaz = Lmz(z*eta, -1/z)   (code column: +z*eta)
    return e.subs(z, -1 / z).subs(m, z * eta)


def etaz_to_mz(e):               # VI: Lmz = Letaz(-z*m, -1/z)
    return e.subs(z, -1 / z).subs(eta, -z * m)


# --- conversion graph -------------------------------------------------------
# edge: (from, to) -> function
_EDGES = {
    ("mz", "gz"): mz_to_gz, ("gz", "mz"): gz_to_mz,
    ("gz", "rg"): gz_to_rg, ("rg", "gz"): rg_to_gz,
    ("mz", "sy"): mz_to_sy, ("sy", "mz"): sy_to_mz,
    ("mz", "muz"): mz_to_muz, ("muz", "mz"): muz_to_mz,
    ("mz", "etaz"): mz_to_etaz, ("etaz", "mz"): etaz_to_mz,
}

# adjacency for BFS shortest path
_ADJ = {}
for (a, b) in _EDGES:
    _ADJ.setdefault(a, []).append(b)


def _path(src, dst):
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
    if target not in ENCODING_VARS:
        raise ValueError(
            f"unknown target encoding {target!r}; known: {list(ENCODING_VARS)}"
        )
    # identify source by its var pair
    src = None
    for name, (v1, v2) in ENCODING_VARS.items():
        if (bp.var1, bp.var2) == (v1, v2):
            src = name
            break
    if src is None:
        raise ValueError(
            f"vars ({bp.var1}, {bp.var2}) do not correspond to any known encoding"
        )
    if src == target:
        return bp.normalize()
    expr = bp.expr
    steps = _path(src, target)
    for a, b in zip(steps, steps[1:]):
        expr = sp.expand(sp.together(_EDGES[(a, b)](expr)))
        # normalize at each hop to keep degrees down and clear denominators;
        # the final hop therefore already yields the canonical target polynomial
        v1, v2 = ENCODING_VARS[b]
        expr = BivariatePolynomial(expr, v1, v2).normalize().expr
    return BivariatePolynomial(expr, *ENCODING_VARS[target]).normalize()
