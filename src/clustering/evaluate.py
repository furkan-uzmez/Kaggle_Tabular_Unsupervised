"""Internal clustering metrics + bootstrap stability + n_clusters scan.

Submission ARI is **not** computed here -- by design, model selection uses
only the signals defined in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass(frozen=True)
class InternalMetrics:
    """Container for the four internal signals.

    `bic` is None for non-probabilistic models.
    """

    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    bic: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _exclude_noise(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows whose label is -1 (HDBSCAN/DBSCAN noise)."""
    mask = labels != -1
    return X[mask], labels[mask]


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    bic: float | None = None,
    exclude_noise: bool = True,
) -> InternalMetrics:
    """Compute silhouette / DB / CH on `X, labels`.

    All three sklearn metrics require at least 2 distinct clusters and at least
    2 samples per cluster. If those preconditions are violated, the metric is
    set to None rather than raising.

    Args:
        X: preprocessed feature matrix.
        labels: cluster labels (use -1 for noise; see `exclude_noise`).
        bic: optional BIC value to attach (GMM/BGMM only).
        exclude_noise: when True, drop label == -1 rows before scoring.
    """
    if exclude_noise:
        X_scored, labels_scored = _exclude_noise(X, labels)
    else:
        X_scored, labels_scored = X, labels

    unique = np.unique(labels_scored)
    if unique.size < 2 or X_scored.shape[0] < unique.size + 1:
        return InternalMetrics(silhouette=None, davies_bouldin=None, calinski_harabasz=None, bic=bic)

    return InternalMetrics(
        silhouette=float(silhouette_score(X_scored, labels_scored)),
        davies_bouldin=float(davies_bouldin_score(X_scored, labels_scored)),
        calinski_harabasz=float(calinski_harabasz_score(X_scored, labels_scored)),
        bic=bic,
    )
