from matrix_library.matrix import Matrix
import math


class PowerIteration:

    def _norm(self, v) -> float:
        return math.sqrt(sum(x * x for x in v))

    def _to_vector(self, matrix: Matrix) -> list:
        if matrix.cols != 1:
            raise ValueError("Not a column vector")
        return [row[0] for row in matrix.data]

    def _to_matrix(self, vector: list) -> Matrix:
        return Matrix([[x] for x in vector])

    def single_eigenvalue(self, matrix: Matrix, iterations=1000, tol=1e-6,seed=0):
        n = len(matrix.data)
        start = [1.0 if i == 0 else 0.1 * (i + seed + 1) for i in range(n)]
        norm = self._norm(start)
        start = [x / norm for x in start]
        v = self._to_matrix(start)

        for _ in range(iterations):
            AV = matrix @ v
            AV_list = self._to_vector(AV)
            norm = self._norm(AV_list)

            if norm < 1e-12:
                break

            v_new = [x / norm for x in AV_list]
            v_old = self._to_vector(v)

            if sum(abs(v_new[i] - v_old[i]) for i in range(n)) < tol:
                v = self._to_matrix(v_new)
                break

            v = self._to_matrix(v_new)

        # v is unit length from loop — Rayleigh quotient simplifies to v^T A v
        AV = matrix @ v
        AV_list = self._to_vector(AV)
        v_list = self._to_vector(v)
        eigenvalue = sum(v_list[i] * AV_list[i] for i in range(n))

        return eigenvalue, v

    def compute(self, matrix: Matrix, iterations=1000, tol=1e-6):
        if matrix.rows != matrix.cols:
            raise ValueError(f"Expected square matrix, got shape {matrix.shape()}")

        eigenvalues = []
        eigenvectors = []
        A = matrix

        for idx in range(matrix.rows):
            eigenvalue, eigenvector = self.single_eigenvalue(A, iterations, tol,seed=idx)
            v_list = self._to_vector(eigenvector)

            if abs(eigenvalue) < 1e-10:
                break

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v_list)

            A = A - eigenvalue * (eigenvector @ eigenvector.transpose())

        return eigenvalues, eigenvectors
