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
