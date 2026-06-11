"""Agglomerative + Spectral clustering wrappers, including a KNN-propagation
helper to scale Spectral to large datasets."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestNeighbors

SPECTRAL_RBF_MAX_ROWS = 20_000
SPECTRAL_SUBSAMPLE_SIZE = 10_000


def agglomerative_factory(
    n_clusters: int,
    *,
    linkage: str = "ward",
):
    """Return a `fit_predict(X) -> labels` closure for AgglomerativeClustering.

    Notes:
        - `linkage='ward'` requires `metric='euclidean'` (sklearn default).
        - Memory cost is O(n^2). For very large n, consider precomputed
          connectivity (out of scope for this plan).
    """

    def _fp(X: np.ndarray) -> np.ndarray:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X)

    return _fp


def spectral_factory(
    n_clusters: int,
    *,
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 10,
    assign_labels: str = "kmeans",
    random_state: int = 42,
    rbf_max_rows: int = SPECTRAL_RBF_MAX_ROWS,
    subsample_size: int = SPECTRAL_SUBSAMPLE_SIZE,
    knn_propagation_k: int = 5,
):
    """Return a `fit_predict(X) -> labels` closure for SpectralClustering.

    Behavior:
        - If `affinity='rbf'` and len(X) > `rbf_max_rows`, fits Spectral on a
          `subsample_size` random subsample then propagates labels to the rest
          via `knn_propagation_k`-NN majority vote.
        - Otherwise uses sklearn Spectral directly.
    """

    def _fp(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if affinity == "rbf" and n > rbf_max_rows:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=min(subsample_size, n), replace=False)
            sub_labels = SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                assign_labels=assign_labels,
                random_state=random_state,
            ).fit_predict(X[idx])
            # Propagate via KNN majority vote on the *features* of the subsample.
            nn = NearestNeighbors(n_neighbors=knn_propagation_k).fit(X[idx])
            _, neighbor_idx = nn.kneighbors(X)
            neighbor_labels = sub_labels[neighbor_idx]
            out = np.empty(n, dtype=int)
            for i, row in enumerate(neighbor_labels):
                vals, counts = np.unique(row, return_counts=True)
                out[i] = int(vals[np.argmax(counts)])
            return out
        else:
            return SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                n_neighbors=n_neighbors,
                assign_labels=assign_labels,
                random_state=random_state,
            ).fit_predict(X)

    return _fp
