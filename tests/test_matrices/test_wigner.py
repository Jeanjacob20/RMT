"""Manual GOE/GUE spectral-distribution demo (run as a script, not a pytest test).

This is a Phase-1 demonstration script, not an automated test. Its plotting code
is guarded under ``__main__`` so that ``pytest`` collects nothing from it (and does
not hang on ``plt.show()``). It is slated for replacement by the real samplers and
viz layer in Phase 3.
"""

from rmtool_py.matrices import RandomMatrixGenerator
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Générer une matrice GOE
    p = 100
    sigma = 1.0
    goe = RandomMatrixGenerator.generate_goe(p, sigma)

    # Générer une matrice GUE
    gue = RandomMatrixGenerator.generate_gue(p, sigma)

    # Afficher les dimensions des matrices
    print("Matrice GOE générée avec succès ! Forme :", goe.shape)
    print("Matrice GUE générée avec succès ! Forme :", gue.shape)

    # Calculer et afficher les valeurs propres
    eigenvalues_goe = np.linalg.eigvalsh(goe)
    eigenvalues_gue = np.linalg.eigvalsh(np.real(gue))  # Les valeurs propres de GUE sont réelles

    # Tracer les distributions spectrales
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(eigenvalues_goe, bins=30, density=True, edgecolor="black", alpha=0.7)
    plt.title("Distribution Spectrale GOE")
    plt.xlabel("Valeur Propre")
    plt.ylabel("Densité")
    plt.xlim(-2.5, 2.5)

    plt.subplot(1, 2, 2)
    plt.hist(eigenvalues_gue, bins=30, density=True, edgecolor="black", alpha=0.7)
    plt.title("Distribution Spectrale GUE")
    plt.xlabel("Valeur Propre")
    plt.ylabel("Densité")
    plt.xlim(-2.5, 2.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
