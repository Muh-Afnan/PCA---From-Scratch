from matrix_library.matrix import Matrix
import math


class StandardScaler:
    def __init__(self):
        self.means = []
        self.stds = []

    def _mean(self, data: list) -> float:
        return sum(data) / len(data)

    def _std(self, mean: float, data: list) -> float:
        return math.sqrt(sum((x - mean) ** 2 for x in data) / (len(data) - 1))

    def fit(self, data: Matrix):
        self.means = []
        self.stds = []

        cols = list(zip(*data.data))

        for col in cols:
            mean = self._mean(col)
            std = self._std(mean, col)

            self.means.append(mean)
            self.stds.append(std)

    def transform(self, data: Matrix) -> Matrix:
        cols = list(zip(*data.data))

        scaled = []

        for i, col in enumerate(cols):
            scaled_col = [
                (x - self.means[i]) / self.stds[i] if self.stds[i] >1e-9 else 0.0
                for x in col
            ]
            scaled.append(scaled_col)

        return Matrix([list(row) for row in zip(*scaled)])

    def fit_transform(self, data: Matrix) -> Matrix:
        self.fit(data)
        return self.transform(data)

    def inverse_transform_scaled(self, data: Matrix) -> Matrix:
        cols = list(zip(*data.data))

        original = []

        for i, col in enumerate(cols):
            original_col = [x * self.stds[i] + self.means[i] for x in col]
            original.append(original_col)

        return Matrix([list(row) for row in zip(*original)])
