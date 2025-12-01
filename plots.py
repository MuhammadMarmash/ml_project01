# dr_and_viz/plots.py

"""
Owner: Abdallah

Goal (MVP):
-----------
Turn coordinates + filenames into nice Plotly figures.

Requirements:
-------------
- 2D scatter (px.scatter)
- 3D scatter (px.scatter_3d)
- Hover must show filename
- Selected document (if any) must be highlighted:
  - different marker size and maybe symbol
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px


def plot_2d(
    coords: np.ndarray,
    filenames: List[str],
    selected_idx: Optional[int],
):
    """
    Build a 2D scatter plot.

    Inputs:
    -------
    coords       : (n_docs, 2) embedding coordinates
    filenames    : list of filenames, same order as coords
    selected_idx : index of selected document or None

    Strategy:
    ---------
    - Build a DataFrame with columns: x, y, filename, selected (bool)
    - Plot non-selected docs as one trace
    - Plot selected doc as another trace with bigger marker / special symbol
    """
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "filename": filenames,
            "selected": [i == selected_idx for i in range(len(filenames))],
        }
    )

    df_non = df[~df["selected"]]
    df_sel = df[df["selected"]]

    # Base scatter: all non-selected documents
    fig = px.scatter(
        df_non,
        x="x",
        y="y",
        hover_name="filename",
        labels={"x": "Component 1", "y": "Component 2"},
    )

    # Add selected document as a separate trace
    if not df_sel.empty:
        fig_sel = px.scatter(
            df_sel,
            x="x",
            y="y",
            hover_name="filename",
        )
        # Make selected point stand out (bigger, star symbol)
        fig_sel.update_traces(marker=dict(size=14, symbol="star"))
        fig.add_traces(fig_sel.data)

    return fig


def plot_3d(
    coords: np.ndarray,
    filenames: List[str],
    selected_idx: Optional[int],
):
    """
    Build a 3D scatter plot.

    Inputs:
    -------
    coords       : (n_docs, 3) embedding coordinates
    filenames    : list of filenames
    selected_idx : index of selected document or None
    """
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "filename": filenames,
            "selected": [i == selected_idx for i in range(len(filenames))],
        }
    )

    df_non = df[~df["selected"]]
    df_sel = df[df["selected"]]

    fig = px.scatter_3d(
        df_non,
        x="x",
        y="y",
        z="z",
        hover_name="filename",
        labels={
            "x": "Component 1",
            "y": "Component 2",
            "z": "Component 3",
        },
    )

    if not df_sel.empty:
        fig_sel = px.scatter_3d(
            df_sel,
            x="x",
            y="y",
            z="z",
            hover_name="filename",
        )
        fig_sel.update_traces(marker=dict(size=10, symbol="diamond"))
        fig.add_traces(fig_sel.data)

    return fig
