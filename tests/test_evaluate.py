"""Sanity tests for the evaluate module on synthetic blobs."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from clustering.evaluate import (
    bootstrap_stability,
    compute_internal_metrics,
    scan_n_clusters,
)


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.6, n_features=4, random_state=0)
    return X, y


def _kmeans_fp(k: int):
    def _inner(X: np.ndarray) -> np.ndarray:
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    return _inner


def test_internal_metrics_on_blobs(blobs):
    X, _ = blobs
    labels = _kmeans_fp(3)(X)
    m = compute_internal_metrics(X, labels)
    assert m.silhouette is not None and m.silhouette > 0.5
    assert m.davies_bouldin is not None and m.davies_bouldin < 1.0
    assert m.calinski_harabasz is not None and m.calinski_harabasz > 100


def test_internal_metrics_degenerate_returns_none():
    X = np.arange(20).reshape(-1, 1).astype(float)
    labels = np.zeros(20, dtype=int)  # single cluster
    m = compute_internal_metrics(X, labels)
    assert m.silhouette is None
    assert m.davies_bouldin is None
    assert m.calinski_harabasz is None


def test_bootstrap_stability_high_on_blobs(blobs):
    X, _ = blobs
    report = bootstrap_stability(_kmeans_fp(3), X, n_iter=10, sample_frac=0.8)
    assert report.n_iter == 10
    assert 0.0 <= report.mean_ari <= 1.0
    assert report.mean_ari > 0.8  # blobs are easy -> high stability


def test_scan_n_clusters_recovers_k_on_blobs(blobs):
    X, _ = blobs

    def model_factory(k: int):
        return _kmeans_fp(k)

    table = scan_n_clusters(
        model_factory,
        X,
        k_range=range(2, 7),
        stability_n_iter=5,
        stability_sample_frac=0.7,
    )
    assert set(table.keys()) == set(range(2, 7))

    # At least one signal should peak / dip at k=3
    sils = {k: row["metrics"].silhouette for k, row in table.items()}
    best_sil_k = max(sils, key=lambda k: sils[k])
    assert best_sil_k == 3
