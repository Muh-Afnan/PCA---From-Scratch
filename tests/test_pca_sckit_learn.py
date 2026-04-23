import unittest
import random
import math
import numpy as np

from matrix_library.matrix import Matrix
from src.pca import PCA as MyPCA
from sklearn.decomposition import PCA as SkPCA


class TestPCAVsSklearn(unittest.TestCase):

    def generate_data(self, rows=300, cols=5, seed=42):
        random.seed(seed)
        return Matrix(
            [[random.uniform(-100, 100) for _ in range(cols)] for _ in range(rows)]
        )

    def matrix_to_numpy(self, M: Matrix):
        return np.array(M.data)

        # ---------- Core comparison ----------

        # def test_projection_similarity(self):
        X = self.generate_data(300, 5)
        X_np = self.matrix_to_numpy(X)

        # --- your PCA ---
        my_pca = MyPCA(n_components=3)
        X_my = my_pca.fit_transform(X)

        # --- sklearn PCA ---
        sk_pca = SkPCA(n_components=3)
        X_sk = sk_pca.fit_transform(X_np)

        X_my_np = self.matrix_to_numpy(X_my)

        # Align scale differences (important)
        def normalize(mat):
            return (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)

        my_norm = normalize(X_my_np)
        sk_norm = normalize(X_sk)
        for col in range(my_norm.shape[1]):
            if np.dot(my_norm[:, col], sk_norm[:, col]) < 0:
                my_norm[:, col] *= -1

        # compare projections
        diff = np.mean(np.abs(my_norm - sk_norm))
        print("My components:\n", my_pca.components_.data)
        print("SK components:\n", sk_pca.components_.T)
        print("My variance ratio:", my_pca.explained_variance_ratio_)
        print("SK variance ratio:", sk_pca.explained_variance_ratio_.tolist())
        print(f"diff: {diff:.4f}")
        self.assertTrue(diff < 0.35, f"diff: {diff:.4f}")

    def test_projection_similarity(self):
        """Use structured data with clearly separated eigenvalues."""
        random.seed(42)
        rows = 300
        data = []
        for _ in range(rows):
            x = random.gauss(0, 10)
            y = random.gauss(0, 5)
            z = random.gauss(0, 0.5)
            a = random.gauss(0, 0.3)
            b = random.gauss(0, 0.2)
            data.append([x, y, z, a, b])

        X = Matrix(data)
        X_np = np.array(data)

        my_pca = MyPCA(n_components=2)
        X_my = my_pca.fit_transform(X)
        X_my_np = self.matrix_to_numpy(X_my)

        # sklearn must use same preprocessing — standardise first
        from sklearn.preprocessing import StandardScaler as SkScaler

        sk_scaler = SkScaler()
        X_sk_scaled = sk_scaler.fit_transform(X_np)
        sk_pca = SkPCA(n_components=2)
        X_sk = sk_pca.fit_transform(X_sk_scaled)

        # align signs column by column
        for col in range(X_my_np.shape[1]):
            if np.dot(X_my_np[:, col], X_sk[:, col]) < 0:
                X_my_np[:, col] *= -1

        # compare correlation of projections (scale-invariant)
        diff = np.mean(
            [
                1 - abs(np.corrcoef(X_my_np[:, c], X_sk[:, c])[0, 1])
                for c in range(X_my_np.shape[1])
            ]
        )
        self.assertTrue(diff < 0.01, f"diff too large: {diff:.4f}")

    # ---------- Variance comparison ----------

    def test_explained_variance_similarity(self):
        X = self.generate_data(400, 6)
        X_np = self.matrix_to_numpy(X)

        my_pca = MyPCA(n_components=6)
        my_pca.fit(X)

        sk_pca = SkPCA(n_components=6)
        sk_pca.fit(X_np)

        my_var = my_pca.explained_variance_
        sk_var = sk_pca.explained_variance_

        # normalize
        my_var = np.array(my_var) / sum(my_var)
        sk_var = sk_var / sum(sk_var)

        diff = np.mean(np.abs(my_var - sk_var))

        self.assertTrue(diff < 0.1)

    # ---------- Reconstruction check ----------

    def test_reconstruction_quality(self):
        X = self.generate_data(200, 4)
        X_np = self.matrix_to_numpy(X)

        # with ALL components, reconstruction must be near-perfect
        my_pca = MyPCA(n_components=4)
        X_t = my_pca.fit_transform(X)
        X_rec = my_pca.inverse_transform(X_t)
        my_rec = self.matrix_to_numpy(X_rec)

        my_err = np.mean((X_np - my_rec) ** 2)
        self.assertTrue(my_err < 1.0, f"Reconstruction error too high: {my_err:.4f}")


if __name__ == "__main__":
    unittest.main()
