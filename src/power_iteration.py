from matrix_library.src.matrix import Matrix
import math

class PowerIteration():
    def _norm(self,v)->"int | float":
        return math.sqrt(sum(x*x for x in v))
    
    def _to_vector(self,matrix:Matrix)->list:
        if matrix.cols != 1:
            raise ValueError("Not a column vector")
        return [row[0] for row in matrix.data]
    
    def _to_matrix(self, vector:list["float | int"])->Matrix:
        return Matrix([[x] for x in vector])
    
    def calculate_eigen(self,matrix:Matrix,iterations=1000, tol=1e-6):
        n = len(matrix.data)

        v = self._to_matrix([1.0]*n)

        for _ in range(iterations):
            AV = matrix @ v
            AV_list = self._to_vector(AV)

            norm = self._norm(AV_list)
            v_new = [x/ norm for x in AV_list]

            v_old = self._to_vector(v)

            if sum(abs(v_new[i]-v_old[i]) for i in range(n))<tol:
                v = self._to_matrix(v_new)
                break
            v = self._to_matrix(v_new)

        AV = matrix @ v
        v_T = v.transpose()
        eigen_value = (v_T @ AV).data[0][0]
        AV_list = self._to_vector(AV)
        v_list = self._to_vector(v)

        eigen_value = sum(v_list[i] * AV_list[i] for i in range(n))

        return eigen_value, v

A = Matrix([[1, 2], [3, 4]])
scalar_obj = PowerIteration()
eigen , vector = scalar_obj.calculate_eigen(A)

print(f"Eigen value is {eigen} and vetor is {vector}")