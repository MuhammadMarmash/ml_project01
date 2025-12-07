# dr.py

import numpy as np
from typing import Literal
from sklearn.decomposition import PCA, TruncatedSVD


def reduce_dimensions(
    X: np.ndarray,
    method: Literal["PCA", "SVD"] = "PCA",
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce high-dimensional document vectors to 2D/3D using sklearn.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (n_docs, n_features). Typically X_filtered (TF-IDF on top keywords).
    method : {"PCA", "SVD"}
        - "PCA": sklearn PCA (centers the data, true principal components).
        - "SVD": sklearn TruncatedSVD (no centering, good for sparse / TF-IDF, LSA-style).
    n_components : int
        Target number of dimensions (2 or 3 for plotting).

    Returns
    -------
    coords : np.ndarray
        Array of shape (n_docs, n_components) with low-dimensional coordinates.
        If the rank is smaller than n_components, remaining columns are zeros.
    """
    if X.size == 0:
        # No features -> return zeros with requested number of components
        return np.zeros((X.shape[0], n_components), dtype=float)

    n_docs, n_features = X.shape

    # We cannot ask for more components than min(n_docs, n_features)
    max_possible = min(n_docs, n_features)
    if max_possible == 0:
        return np.zeros((n_docs, n_components), dtype=float)

    k = min(n_components, max_possible)

    if method == "PCA":
        # sklearn PCA does centering internally
        pca = PCA(n_components=k)
        coords_k = pca.fit_transform(X)  # shape (n_docs, k)

    elif method == "SVD":
        # TruncatedSVD does NOT center; this is typical for TF-IDF / LSA
        # random_state just for reproducibility
        svd = TruncatedSVD(n_components=k, random_state=42)
        coords_k = svd.fit_transform(X)  # shape (n_docs, k)

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    # If k < n_components (rank-deficient case), pad with zeros
    if k == n_components:
        return coords_k.astype(float)

    coords = np.zeros((n_docs, n_components), dtype=float)
    coords[:, :k] = coords_k
    return coords
