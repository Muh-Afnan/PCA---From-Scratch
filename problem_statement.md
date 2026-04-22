# Why This Day Exists

I have already implemented eigenvectors from scratch and built core matrix operations manually.  
Today, I bring those foundations together in a real machine learning algorithm: **PCA (Principal Component Analysis)**.

PCA is one of the most practical algorithms in ML, and I can use it for:

- reducing dimensionality before training models,
- visualizing high-dimensional datasets,
- filtering noise,
- compressing features,
- understanding data structure before modeling.

Most people call `sklearn.decomposition.PCA` and move on.  
In this challenge, I will implement PCA directly from the mathematics, understand its geometric meaning, and validate my implementation against scikit-learn for correctness.
