"""GaussianMixture + BayesianGaussianMixture wrappers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

CovType = Literal["full", "tied", "diag", "spherical"]


def gmm_factory(
    n_components: int,
    *,
    covariance_type: CovType = "full",
    n_init: int = 5,
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    random_state: int = 42,
):
    """Return a `fit_predict(X) -> labels` closure for GaussianMixture.

    Note: GMM `fit_predict` returns hard assignments via argmax of posteriors.
    """

    def _fp(X: np.ndarray) -> np.ndarray:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            reg_covar=reg_covar,
            max_iter=max_iter,
            random_state=random_state,
        )
        return model.fit_predict(X)

    return _fp


def gmm_bic_fn(
    *,
    covariance_type: CovType = "full",
    n_init: int = 5,
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    random_state: int = 42,
):
    """Return a `(k, X) -> bic` closure for use with `scan_n_clusters(bic_fn=...)`.

    Internally re-fits a GMM (no label caching) because BIC must come from the
    same fit you would score. Cheap enough at the n_clusters-scan scale.
    """

    def _bic(k: int, X: np.ndarray) -> float:
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init,
            reg_covar=reg_covar,
            max_iter=max_iter,
            random_state=random_state,
        ).fit(X)
        return float(model.bic(X))

    return _bic


def bgmm_factory(
    n_components: int = 30,
    *,
    covariance_type: CovType = "full",
    weight_concentration_prior_type: str = "dirichlet_process",
    weight_concentration_prior: float = 1e-2,
    max_iter: int = 500,
    reg_covar: float = 1e-6,
    random_state: int = 42,
):
    """Return a `fit_predict(X) -> labels` closure for BayesianGaussianMixture.

    The Dirichlet-process prior lets unused components shrink, effectively
    selecting `k`. Inspect `np.unique(labels).size` after fitting to see how
    many were actually used.
    """

    def _fp(X: np.ndarray) -> np.ndarray:
        model = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        return model.fit_predict(X)

    return _fp
