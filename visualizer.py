"""Render 2D and 3D swing-trajectory visualisations."""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from swing_analyzer import SwingData


def plot_trajectory_2d(data: SwingData) -> Figure:
    df = data.swing_frames.dropna(subset=["wrist_mid_x", "wrist_mid_y"])

    fig, ax = plt.subplots(figsize=(8, 6))
    if data.snapshot_bgr is not None:
        snapshot_rgb = cv2.cvtColor(data.snapshot_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(snapshot_rgb, alpha=0.45, extent=(0, data.width, data.height, 0))

    ax.set_xlim(0, data.width)
    ax.set_ylim(data.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_title("Swing trajectory (2D)")

    if len(df) >= 2:
        xs = df["wrist_mid_x"].to_numpy()
        ys = df["wrist_mid_y"].to_numpy()
        ts = df["t"].to_numpy()
        points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", linewidth=3)
        lc.set_array(ts[:-1])
        ax.add_collection(lc)
        ax.scatter(xs[0], ys[0], c="lime", s=60, zorder=5, label="start", edgecolors="black")
        ax.scatter(xs[-1], ys[-1], c="red", s=60, zorder=5, label="end", edgecolors="black")
        cbar = fig.colorbar(lc, ax=ax, shrink=0.8)
        cbar.set_label("t (s)")
        ax.legend(loc="upper right")
    else:
        ax.text(
            0.5,
            0.5,
            "Not enough wrist detections to draw a trajectory.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    return fig


def plot_trajectory_3d(data: SwingData) -> go.Figure:
    df = data.swing_frames.dropna(subset=["wrist_mid_wx", "wrist_mid_wy", "wrist_mid_wz"])

    fig = go.Figure()
    if len(df) >= 2:
        xs = df["wrist_mid_wx"].to_numpy()
        # MediaPipe world Y points downward; flip so Z-up feels natural.
        ys = -df["wrist_mid_wy"].to_numpy()
        zs = df["wrist_mid_wz"].to_numpy()
        ts = df["t"].to_numpy()

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=zs,
                z=ys,
                mode="lines+markers",
                line=dict(color=ts, colorscale="Viridis", width=6),
                marker=dict(size=3, color=ts, colorscale="Viridis"),
                name="wrist mid",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]],
                y=[zs[0]],
                z=[ys[0]],
                mode="markers",
                marker=dict(size=6, color="lime", line=dict(color="black", width=1)),
                name="start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[xs[-1]],
                y=[zs[-1]],
                z=[ys[-1]],
                mode="markers",
                marker=dict(size=6, color="red", line=dict(color="black", width=1)),
                name="end",
            )
        )
    else:
        fig.add_annotation(text="Not enough 3D landmarks detected.", showarrow=False)

    fig.update_layout(
        title="Swing trajectory (3D, relative to hips)",
        scene=dict(
            xaxis_title="X: left / right (m)",
            yaxis_title="Z: forward / back (m)",
            zaxis_title="Y: up / down (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_speed_profile(data: SwingData) -> Figure:
    df = data.swing_frames

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if df["wrist_speed_m_s"].notna().any():
        ax.plot(df["t"], df["wrist_speed_m_s"], color="tab:blue", label="world speed (m/s)")
        ax.set_ylabel("speed (m/s)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
    if df["wrist_speed_px_s"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(df["t"], df["wrist_speed_px_s"], color="tab:orange", alpha=0.6, label="pixel speed (px/s)")
        ax2.set_ylabel("speed (px/s)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_xlabel("t (s)")
    ax.set_title("Wrist speed profile")
    fig.tight_layout()
    return fig
