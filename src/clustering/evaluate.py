"""Internal clustering metrics + bootstrap stability + n_clusters scan.

Submission ARI is **not** computed here -- by design, model selection uses
only the signals defined in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
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


@dataclass(frozen=True)
class StabilityReport:
    """Bootstrap-stability summary."""

    mean_ari: float
    std_ari: float
    n_iter: int
    sample_frac: float
    per_iter: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


FitPredictFn = Callable[[np.ndarray], np.ndarray]


def bootstrap_stability(
    fit_predict: FitPredictFn,
    X: np.ndarray,
    *,
    n_iter: int = 20,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> StabilityReport:
    """Measure how consistently a clustering procedure labels the same points.

    Protocol:
      1. Fit on full X -> reference labels.
      2. For each of `n_iter` iterations, draw a random subsample of size
         `sample_frac * len(X)` *without replacement* (so indices are unique),
         re-fit, then compare new labels to reference labels on the subsample
         indices via ARI.
      3. Report mean and std of those ARIs.

    Args:
        fit_predict: callable taking X and returning integer cluster labels.
            Use a fresh model instance inside if reseeding matters.
        X: preprocessed feature matrix.
        n_iter: number of bootstrap iterations.
        sample_frac: subsample size as a fraction of N.
        random_state: seeds the index sampler (model reseeding is the caller's
            responsibility via `fit_predict`).

    Returns:
        StabilityReport.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    sub_n = max(2, int(round(sample_frac * n)))

    reference_labels = np.asarray(fit_predict(X))

    aris: list[float] = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=sub_n, replace=False)
        sub_labels = np.asarray(fit_predict(X[idx]))
        ari = float(adjusted_rand_score(reference_labels[idx], sub_labels))
        aris.append(ari)

    arr = np.asarray(aris, dtype=float)
    return StabilityReport(
        mean_ari=float(arr.mean()),
        std_ari=float(arr.std(ddof=0)),
        n_iter=n_iter,
        sample_frac=sample_frac,
        per_iter=aris,
    )


def scan_n_clusters(
    model_factory: Callable[[int], FitPredictFn],
    X: np.ndarray,
    *,
    k_range,
    stability_n_iter: int = 10,
    stability_sample_frac: float = 0.8,
    random_state: int = 42,
    bic_fn: Callable[[int, np.ndarray], float] | None = None,
) -> dict[int, dict]:
    """Sweep n_clusters and record signals for each k.

    Args:
        model_factory: callable `k -> fit_predict(X) -> labels`. The factory
            must produce a deterministic predictor for a given k.
        X: preprocessed feature matrix.
        k_range: iterable of integers (k values to try).
        stability_n_iter: bootstrap iterations per k.
        stability_sample_frac: subsample fraction per bootstrap iteration.
        random_state: seed for bootstrap index sampling.
        bic_fn: optional `(k, X) -> bic` callable for probabilistic models.

    Returns:
        Dict keyed by k. Each value: {"metrics": InternalMetrics,
        "stability": StabilityReport}.
    """
    table: dict[int, dict] = {}
    for k in k_range:
        fp = model_factory(k)
        labels = np.asarray(fp(X))
        bic = bic_fn(k, X) if bic_fn is not None else None
        metrics = compute_internal_metrics(X, labels, bic=bic)
        stability = bootstrap_stability(
            fp,
            X,
            n_iter=stability_n_iter,
            sample_frac=stability_sample_frac,
            random_state=random_state,
        )
        table[int(k)] = {"metrics": metrics, "stability": stability}
    return table
