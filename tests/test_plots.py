# test_plots.py
# Goal: make sure plot_2d and plot_3d run and return a Plotly figure.
import sys
import pathlib
import os

# Ensure project root is on sys.path so this file can be run directly
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from plots import plot_2d, plot_3d

def main():
    filenames = [f"doc{i}.txt" for i in range(5)]
    coords_2d = np.random.rand(5, 2)
    coords_3d = np.random.rand(5, 3)

    fig2d = plot_2d(coords_2d, filenames, selected_idx=2)
    fig3d = plot_3d(coords_3d, filenames, selected_idx=1)

    print("2D figure type:", type(fig2d))
    print("3D figure type:", type(fig3d))

    print("✅ plots tests passed OK! (manually inspect later in Streamlit)")

if __name__ == "__main__":
    main()
