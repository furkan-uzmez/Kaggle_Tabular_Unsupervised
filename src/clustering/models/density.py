"""DBSCAN + HDBSCAN wrappers + noise-handling helpers."""

from __future__ import annotations

import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def k_distance_curve(X: np.ndarray, k: int = 4) -> np.ndarray:
    """Return the sorted distance to the k-th nearest neighbor for each point.

    Used to choose `eps` for DBSCAN: plot this and look for the "knee".
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)  # +1 because self-distance is 0
    dists, _ = nn.kneighbors(X)
    kth = dists[:, k]
    return np.sort(kth)


def dbscan_factory(eps: float, *, min_samples: int = 5, n_jobs: int = -1):
    """Return a `fit_predict(X) -> labels` closure for DBSCAN. Label -1 = noise."""

    def _fp(X: np.ndarray) -> np.ndarray:
        return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit_predict(X)

    return _fp


def hdbscan_factory(
    *,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    core_dist_n_jobs: int = -1,
):
    """Return a `fit_predict(X) -> labels` closure for HDBSCAN. Label -1 = noise."""

    def _fp(X: np.ndarray) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            core_dist_n_jobs=core_dist_n_jobs,
        )
        return clusterer.fit_predict(X)

    return _fp


def assign_noise_to_nearest_cluster(X: np.ndarray, labels: np.ndarray, *, k: int = 5) -> np.ndarray:
    """Reassign label-(-1) points to their majority-vote among k nearest non-noise points.

    If there are no non-noise points, returns labels unchanged.
    """
    labels = labels.copy()
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    clustered_idx = np.where(~noise_mask)[0]
    if clustered_idx.size == 0:
        return labels

    nn = NearestNeighbors(n_neighbors=min(k, clustered_idx.size)).fit(X[clustered_idx])
    _, neighbor_idx = nn.kneighbors(X[noise_mask])
    neighbor_labels = labels[clustered_idx][neighbor_idx]

    # Majority vote per row
    new_labels = np.empty(neighbor_labels.shape[0], dtype=int)
    for i, row in enumerate(neighbor_labels):
        values, counts = np.unique(row, return_counts=True)
        new_labels[i] = int(values[np.argmax(counts)])

    labels[noise_mask] = new_labels
    return labels
