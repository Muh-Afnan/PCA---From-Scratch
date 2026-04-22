import unittest
import random
import math
from matrix_library.matrix import Matrix
from src.scalar import StandardScaler


class TestStandardScalerLarge(unittest.TestCase):

    # ---------- Helpers ----------

    def generate_matrix(self, rows, cols, seed=42) -> Matrix:
        random.seed(seed)
        return Matrix(
            [[random.uniform(-1000, 1000) for _ in range(cols)] for _ in range(rows)]
        )

    def column_stats(self, matrix: Matrix):
        cols = list(zip(*matrix.data))
        means = [sum(col) / len(col) for col in cols]
        stds = [
            math.sqrt(sum((x - m) ** 2 for x in col) / (len(col) - 1))
            for col, m in zip(cols, means)
        ]
        return means, stds

    # ---------- Core Large Tests ----------

    def test_large_matrix_mean_std(self):
        data = self.generate_matrix(1000, 50)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        means, stds = self.column_stats(scaled)

        for m in means:
            self.assertTrue(abs(m) < 1e-2)

        for s in stds:
            self.assertTrue(abs(s - 1.0) < 1e-2)

    def test_large_matrix_inverse(self):
        data = self.generate_matrix(800, 40)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform_scaled(scaled)

        for r1, r2 in zip(data.data, recovered.data):
            for a, b in zip(r1, r2):
                self.assertAlmostEqual(a, b, places=2)

    def test_shape_preserved_large(self):
        rows, cols = 1200, 60
        data = self.generate_matrix(rows, cols)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        self.assertEqual(len(scaled.data), rows)
        self.assertEqual(len(scaled.data[0]), cols)

    # ---------- Edge Stress ----------

    def test_large_constant_columns(self):
        rows, cols = 500, 20
        random.seed(0)

        data = []
        for _ in range(rows):
            row = [random.uniform(-100, 100) for _ in range(cols)]
            row[3] = 7.0
            row[7] = -2.0
            data.append(row)

        data = Matrix(data)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        col3 = [row[3] for row in scaled.data]
        col7 = [row[7] for row in scaled.data]

        self.assertTrue(all(x == 0.0 for x in col3))
        self.assertTrue(all(x == 0.0 for x in col7))

    # ---------- Numerical Stability ----------

    def test_large_extreme_values(self):
        rows, cols = 600, 30

        data = Matrix(
            [[1e12 + random.random() for _ in range(cols)] for _ in range(rows)]
        )

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        means, stds = self.column_stats(scaled)

        for m in means:
            self.assertTrue(abs(m) < 1e-2)

        for s in stds:
            self.assertTrue(abs(s - 1.0) < 1e-2)

    # ---------- Failure Modes ----------

    def test_transform_without_fit(self):
        data = self.generate_matrix(10, 5)
        scaler = StandardScaler()

        with self.assertRaises(Exception):
            scaler.transform(data)

    def test_single_row_failure(self):
        data = Matrix([[1, 2, 3]])
        scaler = StandardScaler()

        with self.assertRaises(ZeroDivisionError):
            scaler.fit(data)

    def test_empty_input_failure(self):
        with self.assertRaises(Exception):
            data = Matrix([])
            scaler = StandardScaler()
            scaler.fit(data)


if __name__ == "__main__":
    unittest.main()
