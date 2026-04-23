# Learnings

## What I Actually Learned (Honest Version)

This document covers what changed in my thinking, not just what the algorithm does.

---

## The Math Became Concrete

I had used PCA before. I knew it "finds directions of maximum variance." But I didn't know *why* those directions are eigenvectors until I implemented it.

The covariance matrix `C = X^T X / (n-1)` encodes how every feature varies with every other feature. The eigenvectors of C are the directions along which C acts most simply — they just get scaled, not rotated. That scaling factor is the eigenvalue. And the eigenvalue is literally the variance of the data projected along that eigenvector direction.

So eigenvectors of the covariance matrix are the principal components because they are the directions where variance is "pure" — uncontaminated by other directions. Everything else follows from this.

---

## Power Iteration with Deflation

My `eigen_2x2` from Day 2 used the characteristic polynomial — solving `det(A - λI) = 0` as a quadratic. For n×n matrices that hits a mathematical wall above 4×4 (Abel's theorem proves no algebraic formula exists for degree 5 and above).

Power iteration is the practical answer. The core insight: multiply any vector by a matrix repeatedly and normalise. The component pointing along the dominant eigenvector grows fastest (by λ₁). Everything else grows slower. Eventually you're pointing almost entirely along the dominant eigenvector.

To get the second eigenvector: deflate the matrix by removing the first component's contribution (`A = A - λ₁ * v₁v₁ᵀ`), then repeat. Each deflation step reveals the next most dominant direction.

The bug I hit: starting from `[1, 1, ..., 1]` is a worst case for symmetric matrices whose eigenvectors have equal components. After deflating out the first eigenvector, the residual matrix has the second eigenvector as its dominant direction — but `[1,1]` has zero projection onto `[1,-1]`, so power iteration stalls. Fix: use a varied starting vector that changes with each deflation step.

---

## Sign Ambiguity Is Real and Matters

This was the most surprising lesson of the day.

A principal component `v` and its negative `-v` are mathematically identical — both are valid eigenvectors. Sklearn makes a specific sign choice (largest absolute value entry is positive). My implementation makes a different choice. Both are correct.

This caused a failing test where I compared projections directly. My projections and sklearn's projections were using opposite signs for some components. The projections looked completely different but were mathematically equivalent.

The fix: compare correlations between corresponding projection columns, not raw values. Correlation is sign-invariant. `corr(Xv, X(-v)) = -1` but `|corr| = 1`, which is the right measure of agreement.

Lesson: whenever comparing eigenvectors or PCA results across implementations, always account for sign ambiguity. This applies to neural network weights, covariance structures, and any other context involving eigenvectors.

---

## The Random Data Trap

My first sklearn comparison test used random uniform data with 5 features. The test failed with `diff = 0.37` even after sign alignment. I assumed power iteration was numerically inaccurate.

It wasn't. The eigenvalues of the random data were `[0.22, 0.21, 0.19, ...]` — nearly identical. When eigenvalues are nearly equal, the corresponding eigenvectors are not unique. Any rotation within that eigenspace is valid. Sklearn chose one rotation, my implementation chose a different one. Both were correct answers to a problem with no unique answer.

Lesson: always test numerical algorithms on problems with well-separated eigenvalues. Degenerate cases (repeated or near-equal eigenvalues) have non-unique solutions and are not appropriate for correctness testing.

---

## Bessel's Correction

The covariance formula uses `n-1` not `n`. This is Bessel's correction — it produces an unbiased estimate of population covariance from a sample. Sklearn uses `n-1` by default. If I had used `n`, my eigenvalues would be consistently slightly smaller than sklearn's and the comparison would fail.

Small formula detail, real consequence.

---

## StandardScaler Must Store Statistics

My first version of StandardScaler recomputed mean and std from whatever data was passed to `transform`. This is wrong for a fundamental reason: if you fit a scaler on training data and then transform test data, you want the test data scaled using the *training* statistics — not its own. Using test statistics would leak information about the test distribution.

The correct design: `fit` computes and stores `means_` and `stds_`. `transform` uses stored values. `fit_transform` calls both. This is the sklearn API pattern for a reason.

---

## What I'd Do Differently

**Use SVD instead of eigendecomposition.** SVD is numerically more stable than the covariance → eigendecomposition pipeline, especially for wide matrices (more features than samples). Sklearn uses SVD internally for exactly this reason. For a future version, `X = U Σ V^T` where V contains the principal components directly — no covariance matrix needed.

**Orthogonality check after deflation.** After computing all components, verify they are mutually orthogonal (dot products ≈ 0). This would have caught the deflation numerical drift issue earlier.

**Better starting vector strategy.** The current `[1/(i+seed+1)]` starting vector works but is ad hoc. A more principled approach is to use a random unit vector with a fixed seed — reproducible but unlikely to be aligned with any eigenvector.