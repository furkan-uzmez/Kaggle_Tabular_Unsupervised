"""Shared plotting helpers used by notebooks.

These functions return the Figure (or Axes) so the notebook can save or
further customize. None of them call `plt.show()` -- let the notebook decide.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_signal_curves(
    table: dict[int, dict],
    title: str = "n_clusters scan",
) -> plt.Figure:
    """Plot silhouette / DB / CH / BIC / stability curves vs. k on one figure.

    `table` is the return value of `evaluate.scan_n_clusters`.
    Missing values (e.g., None silhouettes) are skipped with breaks.
    """
    ks = sorted(table.keys())
    sil = [table[k]["metrics"].silhouette for k in ks]
    db = [table[k]["metrics"].davies_bouldin for k in ks]
    ch = [table[k]["metrics"].calinski_harabasz for k in ks]
    bic = [table[k]["metrics"].bic for k in ks]
    stab_mean = [table[k]["stability"].mean_ari for k in ks]
    stab_std = [table[k]["stability"].std_ari for k in ks]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title)

    panels: list[tuple[plt.Axes, str, list, bool]] = [
        (axes[0, 0], "Silhouette (higher better)", sil, False),
        (axes[0, 1], "Davies-Bouldin (lower better)", db, False),
        (axes[0, 2], "Calinski-Harabasz (higher better)", ch, False),
        (axes[1, 0], "BIC (lower better, GMM only)", bic, False),
        (axes[1, 1], "Stability mean ARI", stab_mean, False),
    ]
    for ax, label, values, _ in panels:
        xs = [k for k, v in zip(ks, values) if v is not None]
        ys = [v for v in values if v is not None]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[1, 2].errorbar(ks, stab_mean, yerr=stab_std, fmt="o-", capsize=3)
    axes[1, 2].set_xlabel("k")
    axes[1, 2].set_ylabel("Stability mean +/- std")
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_2d_clusters(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "2D cluster scatter",
    s: int = 6,
    alpha: float = 0.6,
) -> plt.Figure:
    """Scatter the first two coordinates of `coords_2d`, colored by `labels`.

    Noise points (label == -1) are drawn in gray.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(np.unique(labels).tolist())
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            coords_2d[noise_mask, 0],
            coords_2d[noise_mask, 1],
            c="lightgray",
            s=s,
            alpha=alpha,
            label="noise (-1)",
        )
    for lab in unique:
        if lab == -1:
            continue
        mask = labels == lab
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=s, alpha=alpha, label=f"c{lab}")
    ax.set_title(title)
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    if len(unique) <= 20:
        ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    return fig


def plot_elbow(
    ks: Iterable[int],
    values: Iterable[float],
    *,
    ylabel: str,
    title: str = "Elbow",
) -> plt.Figure:
    """Generic elbow plot for any (k, score) sequence."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), list(values), marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
