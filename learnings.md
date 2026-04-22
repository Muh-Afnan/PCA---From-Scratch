# Learnings from PCA Implementation

## 1. Core Intuition
- PCA finds new axes (principal components) that capture maximum variance.
- The first component explains the largest variance, the second explains the next largest, and so on.
- Components are orthogonal to each other.

## 2. Mathematical Building Blocks
- **Mean centering** is required before PCA.
- **Covariance matrix** captures feature relationships.
- **Eigenvectors** of the covariance matrix define principal directions.
- **Eigenvalues** indicate explained variance along each direction.
- Sorting eigenvalues in descending order gives component order.

## 3. Implementation Insights
- Standard pipeline:
    1. Compute feature-wise mean.
    2. Center data.
    3. Compute covariance matrix.
    4. Perform eigendecomposition (or SVD).
    5. Sort components by eigenvalue magnitude.
    6. Select top `k` components.
    7. Project data to lower dimension.
- SVD is often numerically more stable than direct eigendecomposition.

## 4. Practical Lessons
- Feature scaling matters when features are on different ranges.
- Explained variance ratio helps choose `k`.
- Too few components can lose important information.
- Too many components reduce dimensionality benefits.

## 5. Common Pitfalls
- Forgetting centering leads to incorrect components.
- Using unsorted eigenpairs gives wrong projections.
- Mixing row/column conventions causes shape errors.
- Comparing signs of eigenvectors directly can be misleading (sign can flip).

## 6. Validation Checks
- Confirm transformed shape is `(n_samples, k)`.
- Sum of explained variance ratios should be close to 1 when all components are used.
- Reconstruction error should decrease as `k` increases.

## 7. What This Project Reinforced
- Strong link between linear algebra and machine learning.
- Importance of matrix operations, vector projections, and orthogonality.
- Building PCA from scratch improves understanding beyond library usage.