# Problem Statement

## Why This Day Exists

Day 1 built matrix operations from scratch. Day 2 built eigenvectors from scratch. Day 3 is where those two things collide with a real machine learning algorithm.

PCA — Principal Component Analysis — is one of the most practically useful algorithms in ML. It shows up in dimensionality reduction before training, visualising high-dimensional data, noise filtering, and understanding data structure before modelling. Every serious ML engineer encounters it constantly.

Most people call `sklearn.decomposition.PCA` and move on. The goal here is different: implement PCA from the mathematics up, understand exactly what it's doing geometrically, and validate the result against sklearn to confirm correctness.

## The Core Question

Given a dataset with many features, which directions in the data carry the most variance? Not which individual features — which *directions*, which could be combinations of features. Those directions are the principal components. And they are eigenvectors of the covariance matrix.

## Why Build It From Scratch

After Day 1 and Day 2, I had a working matrix library and a working eigen solver. The natural question was: can I build something real on top of them? PCA is the answer to that question.

Building it from scratch forced me to understand:
- Why data must be centred before computing covariance
- What the covariance matrix actually encodes geometrically
- Why eigenvectors of the covariance matrix are the principal directions
- What Bessel's correction (`n-1`) is and why it matters
- Why eigenvalues directly represent variance along each component
- Why PCA components are only defined up to sign — and what that means practically

The last point turned out to be the most surprising lesson of the day.

## Scope

This implementation covers the full PCA pipeline:
- `StandardScaler` — fit on training data, transform new data with stored statistics
- `covariance_matrix` — exact formula with Bessel's correction
- `PowerIteration` — iterative eigen solver for arbitrary n×n matrices with deflation
- `PCA` — full sklearn-compatible API: `fit`, `transform`, `fit_transform`, `inverse_transform`
- Verified against sklearn on structured data with well-separated eigenvalues