# PCA From Scratch

This project implements **Principal Component Analysis (PCA)** from scratch, without using any machine learning library.

## Overview

PCA is a dimensionality reduction technique used to:

- reduce the number of features
- keep the most important information (maximum variance)
- simplify data visualization and preprocessing

This implementation follows the core PCA steps manually.

## What This Project Does

- Computes the mean of each feature
- Centers the dataset
- Computes the covariance matrix
- Finds eigenvalues and eigenvectors
- Sorts principal components by explained variance
- Projects data onto the top `k` components

## Why This Project

The goal is to understand how PCA works internally instead of calling a built-in function.

## Formula Summary

Given centered data matrix \(X\):

1. Covariance matrix:  
    \[
    C = \frac{1}{n-1}X^T X
    \]

2. Eigen decomposition:  
    \[
    C v = \lambda v
    \]

3. Choose top-\(k\) eigenvectors and project:  
    \[
    Z = X W_k
    \]

## Run

```bash
python main.py
```

> Update the command if your entry file has a different name.

## Notes

- Built for learning and educational purposes.
- No external ML library is used for the PCA logic.
- You can extend this to include explained variance ratio plots and reconstruction error.

---
If you meant “strach”, the correct spelling is **scratch**.
+-+-+-+-+-+
## Implementation Approach (Code-Level)

The PCA pipeline is implemented in a clean, step-by-step matrix workflow:

1. **Input to numeric matrix**  
    Data is treated as a 2D matrix (\(n\_samples \times n\_features\)) so every next step can be done with linear algebra operations.

2. **Feature-wise centering**  
    The mean of each column is computed and subtracted from the dataset.  
    This is essential because PCA assumes zero-centered features before covariance is computed.

3. **Covariance from centered data**  
    Instead of using a high-level ML API, covariance is formed directly with matrix multiplication:
    \[
    C = \frac{1}{n-1}X^T X
    \]
    This keeps the implementation mathematically transparent.

4. **Eigen decomposition for directions of variance**  
    Eigenvalues/eigenvectors are extracted from the covariance matrix.  
    - Eigenvectors = principal directions  
    - Eigenvalues = amount of variance captured by each direction

5. **Manual component ranking**  
    Components are sorted in descending eigenvalue order, then the top \(k\) directions are selected.

6. **Projection to lower dimension**  
    Reduced representation is computed with:
    \[
    Z = XW_k
    \]
    where \(W_k\) contains the selected principal vectors.

## Techniques Used

- **Pure linear algebra implementation** (no PCA wrapper/class from ML libraries)
- **Deterministic ordering** of components by explained variance
- **Vectorized operations** for efficient matrix computation
- **Modular logic** that can be extended to:
  - explained variance ratio
  - scree plot
  - inverse transform / reconstruction error

## Why This Approach Is Strong

- It matches the textbook PCA derivation exactly.
- Each computational step is inspectable and debuggable.
- It builds intuition for covariance structure, orthogonal bases, and variance maximization.
- It creates a solid base for implementing advanced variants later (whitening, incremental PCA, kernel PCA).