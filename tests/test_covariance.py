import unittest
import random
import math
from matrix_library.matrix import Matrix
from src.covariance import covariance_matrix


class TestCovarianceMatrixLarge(unittest.TestCase):

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

    # ---------- Shape Tests ----------

    def test_covariance_shape_large(self):
        data = self.generate_matrix(500, 20)

        cov = covariance_matrix(data)

        self.assertEqual(len(cov.data), 20)
        self.assertEqual(len(cov.data[0]), 20)

    # ---------- Symmetry Test ----------

    def test_covariance_is_symmetric(self):
        data = self.generate_matrix(300, 10)

        cov = covariance_matrix(data)

        for i in range(len(cov.data)):
            for j in range(len(cov.data)):
                self.assertAlmostEqual(cov.data[i][j], cov.data[j][i], places=6)

    # ---------- Variance correctness ----------

    def test_diagonal_is_variance(self):
        data = self.generate_matrix(400, 15)

        cov = covariance_matrix(data)

        cols = list(zip(*data.data))

        for i in range(len(cols)):
            expected_var = self.variance(cols[i])
            self.assertAlmostEqual(cov.data[i][i], expected_var, places=5)

    # ---------- Mean centering correctness ----------

    def test_covariance_zero_mean_property(self):
        data = self.generate_matrix(300, 12)

        cov = covariance_matrix(data)

        # covariance matrix should be stable even if shifted
        shifted = Matrix([[x + 1000 for x in row] for row in data.data])

        cov_shifted = covariance_matrix(shifted)

        for i in range(len(cov.data)):
            for j in range(len(cov.data)):
                self.assertAlmostEqual(cov.data[i][j], cov_shifted.data[i][j], places=5)

    # ---------- Constant column edge case ----------

    def test_constant_feature(self):
        rows, cols = 200, 5

        data = []
        for _ in range(rows):
            row = [random.uniform(-10, 10) for _ in range(cols)]
            row[2] = 7.0  # constant feature
            data.append(row)

        data = Matrix(data)

        cov = covariance_matrix(data)

        # variance of constant column must be ~0
        self.assertAlmostEqual(cov.data[2][2], 0.0, places=6)

        # covariance with others must be ~0
        for i in range(cols):
            self.assertAlmostEqual(cov.data[i][2], 0.0, places=6)
            self.assertAlmostEqual(cov.data[2][i], 0.0, places=6)

    # ---------- Numerical stability ----------

    def test_large_values_stability(self):
        data = Matrix([[1e9 + random.random() for _ in range(10)] for _ in range(200)])

        cov = covariance_matrix(data)

        # should not explode
        for i in range(len(cov.data)):
            for j in range(len(cov.data)):
                self.assertTrue(math.isfinite(cov.data[i][j]))

    # ---------- Small dataset correctness ----------

    def test_manual_small_case(self):
        data = Matrix([[1, 2], [3, 4], [5, 6]])

        cov = covariance_matrix(data)

        # manually known covariance
        self.assertAlmostEqual(cov.data[0][0], 4.0, places=6)
        self.assertAlmostEqual(cov.data[1][1], 4.0, places=6)
        self.assertAlmostEqual(cov.data[0][1], 4.0, places=6)
        self.assertAlmostEqual(cov.data[1][0], 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
