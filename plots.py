from typing import List, Optional
import numpy as np
import plotly.graph_objects as go


def plot_2d(
    coords: np.ndarray,
    filenames: List[str],
    selected_idx: Optional[int],
):
    if coords.shape[1] < 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

    coords = np.asarray(coords)
    n_docs = len(filenames)

    fig = go.Figure()

    all_indices = np.arange(n_docs)

    if selected_idx is not None and 0 <= selected_idx < n_docs:
        unselected_indices = all_indices[all_indices != selected_idx]
    else:
        unselected_indices = all_indices
        selected_idx = None

    if len(unselected_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=coords[unselected_indices, 0],
                y=coords[unselected_indices, 1],
                mode="markers",
                marker=dict(color="blue", size=8),
                text=[filenames[i] for i in unselected_indices],
                hovertemplate="%{text}<extra></extra>",
                name="Documents",
            )
        )

    if selected_idx is not None:
        fig.add_trace(
            go.Scatter(
                x=[coords[selected_idx, 0]],
                y=[coords[selected_idx, 1]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="diamond"),
                text=[filenames[selected_idx]],
                hovertemplate="%{text}<extra></extra>",
                name="Selected Document",
            )
        )

    fig.update_layout(
        title="2D Document Visualization",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend_title="Documents",
    )

    return fig


def plot_3d(
    coords: np.ndarray,
    filenames: List[str],
    selected_idx: Optional[int],
):
    while coords.shape[1] < 3:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

    coords = np.asarray(coords)
    n_docs = len(filenames)

    fig = go.Figure()

    all_indices = np.arange(n_docs)

    if selected_idx is not None and 0 <= selected_idx < n_docs:
        unselected_indices = all_indices[all_indices != selected_idx]
    else:
        unselected_indices = all_indices
        selected_idx = None

    if len(unselected_indices) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=coords[unselected_indices, 0],
                y=coords[unselected_indices, 1],
                z=coords[unselected_indices, 2],
                mode="markers",
                marker=dict(color="blue", size=6),
                text=[filenames[i] for i in unselected_indices],
                hovertemplate="%{text}<extra></extra>",
                name="Documents",
            )
        )

    if selected_idx is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[coords[selected_idx, 0]],
                y=[coords[selected_idx, 1]],
                z=[coords[selected_idx, 2]],
                mode="markers",
                marker=dict(color="red", size=10, symbol="diamond"),
                text=[filenames[selected_idx]],
                hovertemplate="%{text}<extra></extra>",
                name="Selected Document",
            )
        )

    fig.update_layout(
        title="3D Document Visualization",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        legend_title="Documents",
    )

    return fig
