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
