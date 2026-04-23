# Day 3 — PCA from Scratch

Principal Component Analysis implemented from scratch using only pure Python and the matrix library built in Day 1.

## What This Is

PCA is a dimensionality reduction algorithm that finds the directions of maximum variance in a dataset. This implementation covers the full pipeline — from raw data to reduced representation and back — without using any ML library for the core logic.

Validated against scikit-learn on structured data with well-separated eigenvalues.

## File Structure

```
pca_implementation/
├── src/
│   ├── pca.py              # PCA class — fit, transform, fit_transform, inverse_transform
│   ├── scalar.py           # StandardScaler — fit/transform with stored statistics
│   ├── covariance.py       # covariance_matrix function with Bessel's correction
│   └── power_iteration.py  # PowerIteration — iterative eigen solver with deflation
├── tests/
│   ├── test_pca.py
│   ├── test_power_iteration.py
│   ├── test_scaler.py
│   └── test_pca_sklearn.py
├── problem_statement.md
├── approach.md
└── learnings.md
```

## The Pipeline

```
Raw data (n × p)
    ↓  StandardScaler.fit_transform
Scaled data (n × p)
    ↓  covariance_matrix
Covariance matrix (p × p)
    ↓  PowerIteration.compute
Eigenvalues + Eigenvectors
    ↓  sort by eigenvalue, take top k
Principal components (p × k)
    ↓  X_scaled @ components
Reduced data (n × k)
```

## Usage

```python
from src.pca import PCA
from matrix_library.matrix import Matrix

X = Matrix([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2]])

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

print(pca.explained_variance_ratio_)  # fraction of variance per component
```

## Key Design Decisions

**PowerIteration with deflation** — the eigen solver from Day 2 only handled 2×2 matrices (characteristic polynomial). For arbitrary n×n matrices, power iteration with deflation is used: find the dominant eigenvector, subtract its contribution from the matrix, repeat. This is O(n²) per iteration and practical for moderate-sized datasets.

**StandardScaler stores training statistics** — `fit` stores means and stds, `transform` uses stored values. This is essential for correctly scaling test data using training statistics without leaking test distribution information.
**Standardisation vs centring** — PCA mathematically requires only centring (zero mean). This implementation standardises fully (zero mean, unit variance) so features on different scales contribute equally. If your features have meaningful variance differences, consider centring only.

**Sign ambiguity** — PCA components are only defined up to sign. `v` and `-v` are the same principal component. The sklearn comparison test uses correlation (sign-invariant) rather than direct value comparison.

**Bessel's correction** — covariance uses `n-1` denominator to produce an unbiased population estimate, matching sklearn's default.

## Verified Against sklearn

```
Eigenvalues match to 4 decimal places.
Eigenvector dot products with numpy reference: all ±1.000.
Projection correlation with sklearn: > 0.99 on well-separated data.
```

## Dependencies

```
matrix_library  (Day 1 — installed as local package)
numpy           (tests only — for sklearn comparison)
scikit-learn    (tests only — for validation)
```

## Run Tests

```bash
python -m pytest tests/ -v
```