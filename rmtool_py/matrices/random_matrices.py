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
