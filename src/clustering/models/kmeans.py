"""KMeans + MiniBatchKMeans wrappers with a uniform fit_predict signature."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def kmeans_factory(n_clusters: int, *, n_init: int = 20, random_state: int = 42):
    """Return a `fit_predict(X) -> labels` closure for KMeans."""

    def _fp(X: np.ndarray) -> np.ndarray:
        return KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        ).fit_predict(X)

    return _fp


def minibatch_kmeans_factory(
    n_clusters: int,
    *,
    batch_size: int = 4096,
    n_init: int = 10,
    random_state: int = 42,
):
    """Return a `fit_predict(X) -> labels` closure for MiniBatchKMeans."""

    def _fp(X: np.ndarray) -> np.ndarray:
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=n_init,
            random_state=random_state,
        ).fit_predict(X)

    return _fp
