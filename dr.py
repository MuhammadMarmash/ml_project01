# dr_and_viz/dr.py

"""
Owner: Abdallah

Goal (MVP):
-----------
Convert high-dimensional document vectors (X_filtered) into 2D or 3D coordinates
for visualization, using PCA or SVD based on numpy.linalg.svd.

Important:
----------
- We do NOT use sklearn here (assignment restriction).
- For both PCA and SVD modes, we first center the data.
"""

from typing import Literal
import numpy as np


def reduce_dimensions(
    X: np.ndarray,
    method: Literal["PCA", "SVD"] = "PCA",
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce X to n_components dimensions.

    Inputs:
    -------
    X         : (n_docs, n_features) filtered document matrix
    method    : "PCA" or "SVD"
    n_components : 2 or 3

    Steps (high level):
    -------------------
    1) Center X by subtracting the mean of each column.
    2) Compute SVD: X_centered = U * S * V^T
    3) For PCA:
       - We can use U * S as our coordinates (projection on principal components).
    4) For SVD mode:
       - We can reuse the same coords, but we MUST be able to explain that it's
         also a low-rank approximation.

    Output:
    -------
    coords: (n_docs, n_components) array for plotting.
    """
    # 1. Center the data (very important for PCA)
    if X.size == 0:
        # Empty input: return zero coordinates
        return np.zeros((X.shape[0], n_components), dtype=float)

    X_centered = X - X.mean(axis=0, keepdims=True)

    # 2. Compute SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 3. Take first n_components columns of U and scale by S
    #    This gives us principal-component coordinates.
    # If the SVD returns fewer components than requested (e.g. small rank),
    # pad with zeros so the returned coords always have shape (n_docs, n_components).
    n_available = min(U.shape[1], len(S))

    coords = np.zeros((U.shape[0], n_components), dtype=float)
    if n_available > 0:
        take = min(n_components, n_available)
        U_reduced = U[:, :take]
        S_reduced = S[:take]
        coords[:, :take] = U_reduced * S_reduced

    # For this assignment, PCA vs SVD mainly affects how you EXPLAIN it.
    # The math here is the same; you can mention that PCA is basically
    # "SVD on the centered data" and we keep the top components.

    return coords
