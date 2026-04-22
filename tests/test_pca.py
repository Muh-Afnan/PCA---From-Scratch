import unittest
import random
import math
from matrix_library.matrix import Matrix
from src.pca import PCA


class TestPCA(unittest.TestCase):

    # ---------- Helpers ----------

    def generate_matrix(self, rows, cols, seed=42) -> Matrix:
        random.seed(seed)
        return Matrix(
            [[random.uniform(-100, 100) for _ in range(cols)] for _ in range(rows)]
        )

    def mean(self, col):
        return sum(col) / len(col)

    def variance(self, col):
        m = self.mean(col)
        return sum((x - m) ** 2 for x in col) / (len(col) - 1)

    # ---------- Shape preservation ----------

    def test_pca_shape_reduction(self):
        X = self.generate_matrix(500, 10)

        pca = PCA(n_components=3)
        X_t = pca.fit_transform(X)

        self.assertEqual(len(X_t.data), 500)
        self.assertEqual(len(X_t.data[0]), 3)

    # ---------- Variance ordering ----------

    def test_explained_variance_sorted(self):
        X = self.generate_matrix(400, 6)

        pca = PCA(n_components=6)
        pca.fit(X)

        ev = pca.explained_variance_

        for i in range(len(ev) - 1):
            self.assertTrue(ev[i] >= ev[i + 1])

    # ---------- Explained variance ratio sanity ----------

    def test_explained_variance_ratio_bounds(self):
        X = self.generate_matrix(300, 5)

        pca = PCA(n_components=5)
        pca.fit(X)

        ratios = pca.explained_variance_ratio_

        self.assertTrue(all(0 <= r <= 1 for r in ratios))
        self.assertTrue(abs(sum(ratios) - 1.0) < 1e-2)

    # ---------- Reconstruction accuracy ----------

    def test_inverse_transform_reconstruction(self):
        X = self.generate_matrix(200, 4)

        pca = PCA(n_components=4)
        X_t = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_t)

        for r1, r2 in zip(X.data, X_recon.data):
            for a, b in zip(r1, r2):
                self.assertAlmostEqual(a, b, places=1)

    # ---------- Dimensionality reduction effect ----------

    def test_variance_reduction_effect(self):
        X = self.generate_matrix(300, 8)

        pca = PCA(n_components=2)
        X_t = pca.fit_transform(X)

        self.assertEqual(len(X_t.data[0]), 2)

    # ---------- Constant feature behavior ----------

    def test_constant_feature_behavior(self):
        rows, cols = 300, 5

        data = []
        for _ in range(rows):
            row = [random.uniform(-10, 10) for _ in range(cols)]
            row[2] = 7.0  # constant feature
            data.append(row)

        X = Matrix(data)

        pca = PCA(n_components=3)
        pca.fit(X)

        # constant feature should contribute almost zero variance
        self.assertTrue(min(pca.explained_variance_) >= 0)

    # ---------- Numerical stability ----------

    def test_large_values_stability(self):
        X = Matrix([[1e9 + random.random() for _ in range(5)] for _ in range(300)])

        pca = PCA(n_components=2)
        X_t = pca.fit_transform(X)

        for row in X_t.data:
            for val in row:
                self.assertTrue(math.isfinite(val))

    # ---------- Fit-Transform consistency ----------

    def test_fit_transform_consistency(self):
        X = self.generate_matrix(200, 6)

        pca = PCA(n_components=3)

        X1 = pca.fit_transform(X)
        X2 = pca.transform(X)

        self.assertEqual(X1.data, X2.data)

    # ---------- Transform without fit ----------

    def test_transform_without_fit(self):
        X = self.generate_matrix(10, 3)

        pca = PCA(n_components=2)

        with self.assertRaises(Exception):
            pca.transform(X)


if __name__ == "__main__":
    unittest.main()
