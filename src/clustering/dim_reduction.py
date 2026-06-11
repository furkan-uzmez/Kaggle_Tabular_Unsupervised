"""PCA and UMAP helpers."""

from __future__ import annotations

import numpy as np
import umap
from sklearn.decomposition import PCA


def fit_pca(X: np.ndarray, n_components: int, *, random_state: int = 42) -> tuple[np.ndarray, PCA]:
    """Fit PCA and return (transformed, fitted_model)."""
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca


def pca_variance_curve(X: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    """Return cumulative explained variance ratio across all components."""
    pca = PCA(random_state=random_state).fit(X)
    return np.cumsum(pca.explained_variance_ratio_)


def fit_umap(
    X: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, "umap.UMAP"]:
    """Fit UMAP and return (transformed, fitted_model).

    Note: UMAP is non-deterministic across hardware even with a seed; small
    drift is expected. For experiment logs, treat `random_state` as a hint.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    Z = reducer.fit_transform(X)
    return Z, reducer
