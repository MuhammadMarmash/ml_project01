# test_dr.py
# Goal: make sure PCA/SVD dimensionality reduction returns correct shapes.
import sys
import pathlib
import os

# Ensure project root is on sys.path so this file can be run directly
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from dr import reduce_dimensions

def main():
    # Fake data: 5 docs, 10 features
    np.random.seed(0)
    X = np.random.rand(5, 10)

    coords_2d = reduce_dimensions(X, method="PCA", n_components=2)
    coords_3d = reduce_dimensions(X, method="PCA", n_components=3)

    print("2D coords shape:", coords_2d.shape)
    print("3D coords shape:", coords_3d.shape)

    assert coords_2d.shape == (5, 2)
    assert coords_3d.shape == (5, 3)

    print("2D coords example:", coords_2d)
    print("3D coords example:", coords_3d)

    print("✅ DR tests passed OK!")

if __name__ == "__main__":
    main()
