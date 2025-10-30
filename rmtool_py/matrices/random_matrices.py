import numpy as np
from scipy.stats import wishart, invwishart, norm


class RandomMatrixGenerator:
    """
    A class to generate random matrices commonly used in random matrix theory.
    Methods:
    --------
    generate_goe(p: int, sigma: float = 1.0) -> np.ndarray:
        Generate a Gaussian Orthogonal Ensemble (GOE) matrix.
    generate_gue(p: int, sigma: float = 1.0) -> np.ndarray:
        Generate a Gaussian Unitary Ensemble (GUE) matrix.
    generate_wishart(p: int, n: int, scale: float = 1.0) -> np.ndarray:
        Generate a Wishart matrix.
    generate_inverse_wishart(p: int, df: int, scale: float = 1.0) -> np.ndarray:
        Generate an Inverse Wishart matrix.
    Notes:
    -------
    GOE and GUE are already normalized to guaranty its convergence to Semicircle law.
    """
    @staticmethod
    def generate_goe(p: int, sigma: float = 1.0) -> np.ndarray:
        """
        Generate a p x p Gaussian Orthogonal Ensemble (GOE) matrix.

        Parameters:
        -----------
        p : int
            Dimension of the matrix (p x p).
        sigma : float, optional
            Scale parameter (default: 1.0).

        Returns:
        --------
        np.ndarray
            A GOE matrix of shape (p, p).
        """
        A = np.zeros((p, p))
        # Fill lower triangle (i > j) with iid standard Gaussian
        idx_lower = np.tril_indices(p, k=-1)
        A[idx_lower] = norm.rvs(loc=0, scale=1, size=len(idx_lower[0]))
        # Fill diagonal (i == j) with N(0,2)
        idx_diag = np.diag_indices(p)
        A[idx_diag] = norm.rvs(loc=0, scale=np.sqrt(2), size=p)
        # Make symmetric
        A = A + A.T - np.diag(np.diag(A))
        return (A / np.sqrt(p))

    @staticmethod
    def generate_gue(p: int, sigma: float = 1.0) -> np.ndarray:
        """
        Generate a p x p Gaussian Unitary Ensemble (GUE) matrix.

        Parameters:
        -----------
        p : int
            Dimension of the matrix (p x p).
        sigma : float, optional
            Scale parameter (default: 1.0).

        Returns:
        --------
        np.ndarray
            A GUE matrix of shape (p, p).
        """
        # Initialize real and imaginary parts
        A_real = np.zeros((p, p))
        A_imag = np.zeros((p, p))

        # Fill lower triangle (i > j) with iid standard Gaussian for real and imaginary parts
        idx_lower = np.tril_indices(p, k=-1)
        A_real[idx_lower] = norm.rvs(loc=0, scale=1/np.sqrt(2), size=len(idx_lower[0]))
        A_imag[idx_lower] = norm.rvs(loc=0, scale=1/np.sqrt(2), size=len(idx_lower[0]))

        # Fill diagonal (i == j) with N(0, 1) for real part only
        idx_diag = np.diag_indices(p)
        A_real[idx_diag] = norm.rvs(loc=0, scale=1, size=p)

        # Make symmetric for real part and anti-symmetric for imaginary part
        A_real = A_real + A_real.T - np.diag(np.diag(A_real))
        A_imag = A_imag - A_imag.T

        # Combine real and imaginary parts
        A = A_real + 1j * A_imag
        return A / np.sqrt(p)

    @staticmethod
    def generate_wishart(p: int, n: int, scale: float = 1.0) -> np.ndarray:
        """
        Generate a p x p Wishart matrix with n degrees of freedom and scale matrix `scale * I`.

        Parameters:
        -----------
        p : int
            Dimension of the matrix (p x p).
        n : int
            Degrees of freedom (must be >= p).
        scale : float, optional
            Scale parameter (default: 1.0).

        Returns:
        --------
        np.ndarray
            A Wishart-distributed random matrix.
        """
        if n < p:
            raise ValueError("Degrees of freedom `n` must be >= dimension `p`.")
        scale_matrix = scale * np.eye(p)

        # Générer la matrice de Wishart
        wishart_matrix = wishart.rvs(df=n, scale=scale_matrix)

        # Normaliser par n pour obtenir une matrice de covariance empirique
        return wishart_matrix / n

    @staticmethod
    def generate_inverse_wishart(p: int, df: int, scale: float = 1.0) -> np.ndarray:
        """
        Generate a p x p Inverse Wishart matrix with df degrees of freedom and scale matrix
        `scale * I`.

        Parameters:
        -----------
        p : int
            Dimension of the matrix (p x p).
        df : int
            Degrees of freedom (must be >= p).
        scale : float, optional
            Scale parameter (default: 1.0).

        Returns:
        --------
        np.ndarray
            An Inverse Wishart-distributed random matrix.
        """
        if df < p:
            raise ValueError("Degrees of freedom `df` must be >= dimension `p`.")

        scale_matrix = scale * np.eye(p)
        return invwishart.rvs(df=df, scale=scale_matrix)
