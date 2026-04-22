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

    def test_projection_similarity(self):
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

        # compare projections
        diff = np.mean(np.abs(my_norm - sk_norm))

        self.assertTrue(diff < 0.2)

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

        my_pca = MyPCA(n_components=4)
        X_t = my_pca.fit_transform(X)
        X_rec = my_pca.inverse_transform(X_t)

        sk_pca = SkPCA(n_components=4)
        X_sk_t = sk_pca.fit_transform(X_np)
        X_sk_rec = sk_pca.inverse_transform(X_sk_t)

        my_rec = self.matrix_to_numpy(X_rec)

        # compare reconstruction error
        my_err = np.mean((X_np - my_rec) ** 2)
        sk_err = np.mean((X_np - X_sk_rec) ** 2)

        # your PCA should not be drastically worse
        self.assertTrue(my_err < sk_err * 5)


if __name__ == "__main__":
    unittest.main()
