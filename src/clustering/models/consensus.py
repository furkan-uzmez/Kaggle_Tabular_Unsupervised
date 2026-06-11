"""Consensus clustering via co-association matrix.

Given N label sets over the same rows, build M[i,j] = fraction of label sets
where i and j share a cluster; then cluster on `1 - M`.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def co_association_matrix(label_sets: list[np.ndarray]) -> np.ndarray:
    """Compute the N x N co-association matrix from a list of label vectors.

    Each label vector must have the same length (n_rows). The result has
    entries in [0, 1], where M[i, j] = (# label sets where i, j co-cluster) / N.
    Diagonal is always 1.

    Implementation note: builds per-label-set binary co-occurrence with broadcasting.
    For very large n (e.g., 100k), this needs O(n^2) memory per accumulation.
    """
    if not label_sets:
        raise ValueError("label_sets must be non-empty")
    n = label_sets[0].shape[0]
    for arr in label_sets:
        if arr.shape[0] != n:
            raise ValueError("All label vectors must have the same length")

    M = np.zeros((n, n), dtype=np.float32)
    for labels in label_sets:
        labels = np.asarray(labels).reshape(-1, 1)
        # Note: -1 noise points only co-cluster with other -1 points, which is
        # not meaningful. We treat each -1 as its own unique cluster id by
        # offsetting them.
        adj_labels = labels.copy()
        noise_mask = adj_labels.flatten() == -1
        if noise_mask.any():
            adj_labels[noise_mask, 0] = np.arange(noise_mask.sum(), dtype=labels.dtype) + (
                labels.max() + 1
            )
        M += (adj_labels == adj_labels.T).astype(np.float32)
    M /= len(label_sets)
    return M


def consensus_labels(
    label_sets: list[np.ndarray],
    n_clusters: int,
    *,
    linkage: str = "average",
) -> np.ndarray:
    """Apply Agglomerative clustering on `1 - co_association_matrix`.

    Uses a precomputed distance matrix (`metric='precomputed'`); only the
    linkages that accept precomputed distances (`average`, `complete`,
    `single`) are valid here. `ward` is NOT supported with precomputed.
    """
    if linkage == "ward":
        raise ValueError("ward linkage requires Euclidean distances; use 'average' instead.")
    M = co_association_matrix(label_sets)
    distance = 1.0 - M
    np.fill_diagonal(distance, 0.0)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage)
    return model.fit_predict(distance)
