# Approach

## Starting Point

Days 1 and 2 produced a working matrix library and a 2×2 eigen solver. The natural question: can I build a real ML algorithm on top of them? PCA is the answer — it requires matrix multiplication, covariance computation, and eigendecomposition, all of which were already built.

The constraint: no sklearn or numpy for the core PCA logic. Only the Day 1 matrix library and pure Python.

## File Decomposition

The pipeline was split into four independent files, each with one responsibility:

**`covariance.py`** — a single function, not a class. It takes a Matrix, returns the p×p covariance matrix with Bessel's correction. No state, no side effects. A class would be unnecessary wrapping for a stateless operation.

**`scalar.py`** — `StandardScaler` class because it has state: it stores `means_` and `stds_` from `fit` so that `transform` can use training statistics on new data. The fit/transform separation is not API decoration — it encodes a real constraint about data leakage.

Note: this implementation standardises (subtracts mean AND divides by std) rather than only centring. This equalises feature scales at the cost of removing natural variance differences between features. sklearn's PCA only centres by default.

**`power_iteration.py`** — `PowerIteration` class implementing the iterative eigen solver with deflation. This replaces the Day 2 `eigen_2x2` which was limited to 2×2 via the characteristic polynomial. The class design groups the helper methods (`_norm`, `_to_vector`, `_to_matrix`) with the algorithm that uses them.

**`pca.py`** — `PCA` class with sklearn-compatible API: `fit`, `transform`, `fit_transform`, `inverse_transform`. It orchestrates the other three components. The sklearn API compatibility was a deliberate choice — it means the implementation can be swapped into any existing sklearn pipeline for comparison.

## Why Power Iteration Over Analytical Methods

The Day 2 solver used `det(A - λI) = 0` as a quadratic formula. This works for 2×2 but the pattern breaks immediately:
- 3×3 requires Cardano's cubic formula — complex and numerically unstable
- 4×4 requires the quartic formula
- 5×5 and above: Abel's theorem proves no algebraic formula exists

Power iteration sidesteps the formula entirely. The geometric insight: multiply any vector by a matrix repeatedly and normalise. The component pointing along the dominant eigenvector grows fastest. After enough iterations, only that component remains.

Deflation extracts subsequent eigenvectors: after finding `v₁` with eigenvalue `λ₁`, subtract `λ₁ * v₁v₁ᵀ` from the matrix. This removes `v₁`'s contribution. The next power iteration finds `v₂`.

## The Starting Vector Problem

A subtle bug: starting from `[1, 1, ..., 1]` is a worst case for symmetric matrices. For `[[2,1],[1,2]]`, the eigenvectors are `[1,1]/√2` and `[1,-1]/√2`. The starting vector `[1,1]` is perfectly aligned with the first eigenvector. After deflation, the residual matrix has `[1,-1]` as its dominant direction. But starting again from `[1,1]` gives zero projection onto `[1,-1]` — power iteration has nothing to amplify and produces garbage.

Fix: use a varied starting vector that changes with each deflation step — `[1/(i+seed+1) for i in range(n)]` where `seed` is the iteration index. This ensures a non-zero projection onto the remaining eigenvectors regardless of their direction.

## The Sign Ambiguity Discovery

The sklearn comparison test initially failed with `diff = 0.37` even after sign-flipping each column. Investigation showed:

1. Eigenvalues matched numpy exactly (4 decimal places)
2. Eigenvectors matched numpy exactly (dot products all ±1.0)
3. The random test data had nearly equal eigenvalues (`0.22, 0.21, 0.19`)

When eigenvalues are nearly equal, eigenvectors are not unique — any rotation within the eigenspace is valid. Sklearn chose one rotation, this implementation chose another. Both correct, both different.

Fix: use structured data with well-separated eigenvalues for comparison tests. Use correlation rather than direct value comparison — correlation is sign-invariant and scale-invariant.

## `inverse_transform` Design

`inverse_transform` must reverse both the projection and the scaling:

```
X_transformed → X_scaled_reconstructed = X_transformed @ components.T
X_scaled_reconstructed → X_original = X * stds + means
```

The second step requires the scaler to be stored from `fit`. This is why `self.scaler_` is an instance variable — `inverse_transform` needs the same statistics that were used to scale the data originally.

## What Was Not Implemented

**SVD-based PCA** — sklearn uses SVD internally because it is numerically more stable than the covariance → eigendecomposition pipeline, particularly for wide matrices. A future version should implement `X = U Σ V^T` where V contains the principal components directly, skipping covariance computation entirely.

**Incremental PCA** — for datasets too large to fit in memory, PCA can be computed in minibatches. Out of scope for Day 3 but a natural extension.

**Kernel PCA** — nonlinear dimensionality reduction by implicitly mapping to a higher-dimensional space. Requires the kernel trick — a Day 15+ topic.