"""Laloux–Bouchaud noise-dressing on real S&P 500 data (WRDS / CRSP).

Reproduces Laloux–Cizeau–Bouchaud–Potters (1999) on a modern sample: fit the
empirical correlation-matrix spectrum to Marčenko–Pastur and show that the
overwhelming majority of eigenvalues fall inside the random-noise band.

Data:  413 stocks continuously in the S&P 500, daily returns 2018–2022
       (CRSP `dsf` returns + `dsp500list` membership), cached as an N×T panel.

Run `pull_sp500.py` first to produce the cached panel, then this script.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rmtool_py.finance import correlation_matrix, fit_marchenko_pastur, mp_edges
from rmtool_py.viz import plot_spectrum

PANEL = "/tmp/sp500_panel.npz"
OUT = "docs/figures/sp500_mp_spectrum.png"

d = np.load(PANEL)
R = d["returns"]                       # N x T daily returns
N, T = R.shape

corr = correlation_matrix(R, standardize=True)   # unit-diagonal correlation matrix
C, Q = corr.C, corr.Q
eigs = np.linalg.eigvalsh(C)
fit = fit_marchenko_pastur(eigs, Q)

s2 = fit.sigma2_market                  # robust noise level (1 - lambda_max/N)
lo, hi = mp_edges(Q, s2)
n_noise = int(np.sum(eigs <= hi))

print(f"N={N}  T={T}  Q={Q:.2f}")
print(f"lambda_max = {eigs[-1]:.1f}  ({100*eigs[-1]/N:.0f}% of total variance)")
print(f"sigma^2 (market) = {s2:.3f}   sigma^2 (lsq) = {fit.sigma2_lsq:.3f}")
print(f"noise band upper edge lambda+ = {hi:.3f}")
print(f"eigenvalues at/below lambda+ = {n_noise}/{N} = {100*n_noise/N:.1f}%")

bulk = eigs[eigs <= 3.0]                 # clip giant market/sector modes for readability
fig, ax = plot_spectrum(bulk, Q=Q, sigma2=s2, bins=60)
ax.axvline(hi, color="crimson", ls="--", lw=1.3,
           label=r"$\lambda_+=%.2f$ (noise edge)" % hi)
ax.set_xlim(0, 3.0)
ax.set_xlabel(r"eigenvalue  $\lambda$")
ax.set_ylabel(r"density  $\rho(\lambda)$")
ax.set_title("S&P 500 correlation spectrum vs Marchenko-Pastur\n"
             f"N={N}, T={T} (2018-2022), Q={Q:.2f} : "
             f"{100*n_noise/N:.0f}% of eigenvalues are noise", fontsize=11)
ax.text(0.97, 0.95,
        f"$\\sigma^2={s2:.2f}$  ($1-\\lambda_{{max}}/N$)\n"
        f"$\\lambda_{{max}}={eigs[-1]:.0f}$ = {100*eigs[-1]/N:.0f}% of variance\n"
        f"(market mode, off-scale)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
