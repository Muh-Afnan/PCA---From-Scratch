import unittest
import math
from matrix_library.matrix import Matrix
from src.power_iteration import PowerIteration


class TestPowerIteration(unittest.TestCase):

    # ---------- Helpers ----------

    def approx(self, a, b, tol=1e-3):
        return abs(a - b) < tol

    def is_unit_vector(self, v, tol=1e-3):
        norm = math.sqrt(sum(x * x for x in v))
        return abs(norm - 1.0) < tol

    # ---------- Core Eigenvalue Test ----------

    def test_simple_diagonal_matrix(self):
        # Eigenvalues should be diagonal elements
        A = Matrix([[3, 0], [0, 2]])

        solver = PowerIteration()
        eigenvalues, eigenvectors = solver.compute(A)

        self.assertEqual(len(eigenvalues), 2)

        # largest eigenvalue first (power iteration behavior)
        self.assertTrue(self.approx(eigenvalues[0], 3.0))
        self.assertTrue(self.approx(eigenvalues[1], 2.0))

    # ---------- Dominant Eigenvalue ----------

    def test_dominant_eigenvalue(self):
        A = Matrix([[4, 1], [2, 3]])

        solver = PowerIteration()
        eigenvalues, _ = solver.compute(A)

        # dominant eigenvalue ~5
        self.assertTrue(any(self.approx(ev, 5.0) for ev in eigenvalues))

    # ---------- Symmetric Matrix Stability ----------

    def test_symmetric_matrix(self):
        A = Matrix([[2, 1], [1, 2]])

        solver = PowerIteration()
        eigenvalues, eigenvectors = solver.compute(A)

        # eigenvalues should be ~3 and ~1
        self.assertTrue(any(self.approx(ev, 3.0) for ev in eigenvalues))
        self.assertTrue(any(self.approx(ev, 1.0) for ev in eigenvalues))

    # ---------- Eigenvector Normalization ----------

    def test_eigenvector_normalization(self):
        A = Matrix([[5, 0], [0, 1]])

        solver = PowerIteration()
        _, eigenvectors = solver.compute(A)

        for v in eigenvectors:
            self.assertTrue(self.is_unit_vector(v))

    # ---------- Convergence Behavior ----------

    def test_convergence_stability(self):
        A = Matrix([[6, 2], [2, 3]])

        solver = PowerIteration()
        eigenvalues, _ = solver.compute(A, iterations=500)

        # should converge to stable values
        self.assertEqual(len(eigenvalues), 2)
        self.assertTrue(max(eigenvalues) > min(eigenvalues))

    # ---------- Edge Case: Identity Matrix ----------

    def test_identity_matrix(self):
        A = Matrix([[1, 0], [0, 1]])

        solver = PowerIteration()
        eigenvalues, _ = solver.compute(A)

        for ev in eigenvalues:
            self.assertTrue(self.approx(ev, 1.0))

    # ---------- Failure Case: Non-square ----------

    def test_non_square_matrix_fails(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])

        solver = PowerIteration()

        with self.assertRaises(Exception):
            solver.compute(A)

    # ---------- Deflation sanity ----------

    def test_deflation_reduces_structure(self):
        A = Matrix([[3, 1], [1, 3]])

        solver = PowerIteration()
        eigenvalues, _ = solver.compute(A)

        # after deflation, second eigenvalue should still be valid
        self.assertTrue(any(self.approx(ev, 4.0) for ev in eigenvalues))
    
    def test_eigenvector_property(self):
        """A @ v must equal λ * v for every eigenvector."""
        A = Matrix([[3, 1], [1, 3]])
        solver = PowerIteration()
        eigenvalues, eigenvectors = solver.compute(A)
        for lam, v_list in zip(eigenvalues, eigenvectors):
            v = Matrix([[x] for x in v_list])
            Av = A @ v
            lv = Matrix([[x * lam] for x in v_list])
            for r1, r2 in zip(Av.data, lv.data):
                self.assertAlmostEqual(r1[0], r2[0], places=3)


if __name__ == "__main__":
    unittest.main()
