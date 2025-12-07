from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px


def plot_2d(
    coords: np.ndarray,
    filenames: List[str],
    selected_idx: Optional[int],
):
    if coords.shape[1] < 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

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

    if not df_non.empty:
        fig = px.scatter(
            df_non,
            x="x",
            y="y",
            hover_name="filename",
            labels={"x": "Component 1", "y": "Component 2"},
        )
    else:
        fig = px.scatter(
            df_sel,
            x="x",
            y="y",
            hover_name="filename",
            labels={"x": "Component 1", "y": "Component 2"},
        )
        fig.update_traces(marker=dict(size=14, symbol="star"))
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
        return fig

    if not df_sel.empty:
        fig_sel = px.scatter(
            df_sel,
            x="x",
            y="y",
            hover_name="filename",
        )
        fig_sel.update_traces(marker=dict(size=14, symbol="star"))
        fig.add_traces(fig_sel.data)

    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def plot_3d(
    coords: np.ndarray,
    filenames: List[str],
    selected_idx: Optional[int],
):
    while coords.shape[1] < 3:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

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

    if not df_non.empty:
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
    else:
        fig = px.scatter_3d(
            df_sel,
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
        fig.update_traces(marker=dict(size=10, symbol="diamond"))
        fig.update_layout(scene=dict(aspectmode="cube"))
        return fig

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

    fig.update_layout(scene=dict(aspectmode="cube"))
    return fig
