from covariance import covariance_matrix
from matrix_library.matrix import Matrix
from scalar import StandardScaler
from power_iteration import PowerIteration


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute the covariance matrix
        cov_matrix = covariance_matrix(X_scaled)

        # Perform power iteration to find eigenvalues and eigenvectors
        power_iter = PowerIteration()
        eigenvalues, eigenvectors = power_iter.compute(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
        sorted_eigenvalues = [eigenvalues[i] for i in sorted_indices]
        sorted_eigenvectors = [eigenvectors[i] for i in sorted_indices]

        # Select the top n_components eigenvectors
        if self.n_components is not None:
            sorted_eigenvectors = sorted_eigenvectors[:self.n_components]
            sorted_eigenvalues = sorted_eigenvalues[:self.n_components]

        self.components_ = Matrix(sorted_eigenvectors).transpose()
        self.explained_variance_ = sorted_eigenvalues

    def transform(self, X):
        if self.components_ is None:
            raise ValueError("PCA has not been fitted yet.")

        # Project the data onto the principal components
        return X_scaled @ self.components_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
