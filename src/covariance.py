from matrix_library.matrix import Matrix
import math


def covariance_matrix(data: Matrix) -> Matrix:
    """Calculate the covariance matrix of a list of lists of numbers.

    Args:
        data (list): A list of lists of numbers.
    Returns:
        list: A covariance matrix as a list of lists.
    """

    # if data.data.rows < 2:
    #     raise ValueError("Need at least 2 samples to compute covariance.")

    features = list(zip(*data.data))

    center = []
    for feature in features:
        mean = sum(feature) / len(feature)
        center.append([x - mean for x in feature])

    cov_matrix = []
    for i in range(len(center)):
        row = []
        for j in range(len(center)):
            single_cov = sum(
                center[i][k] * center[j][k] for k in range(len(center[i]))
            ) / (len(center[i]) - 1)
            row.append(single_cov)
        cov_matrix.append(row)

    return Matrix(cov_matrix)


mat = [
    [2.5, 2.4, 1.7],
    [0.5, 0.7, 1.9],
    [2.2, 2.9, 2.1],
    [1.9, 2.2, 1.8],
    [3.1, 3.0, 2.3],
    [2.3, 2.7, 2.0],
    [2, 1.6, 1.5],
    [1, 1.1, 8],
    [1.5, 1.6, 8],
    [1.1, 0.9, 9],
]
mat = Matrix(mat)
cov_matrix = covariance_matrix(mat)
print(cov_matrix)

