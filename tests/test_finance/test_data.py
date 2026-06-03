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
