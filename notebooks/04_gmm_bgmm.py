# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (Kaggle Tabular Unsupervised)
#     language: python
#     name: kaggle-tabular-unsupervised
# ---

# %% [markdown]
# # 04 -- GaussianMixture + BayesianGaussianMixture
#
# Use the "GMM-friendly" preprocess (typically standard or quantile + yeo-johnson).
# Sweep covariance_type and n_components. BIC is the primary signal.

# %%
import os
import random
import warnings
from pathlib import Path

if Path.cwd().name == "notebooks":
    os.chdir("..")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from clustering.data import load_raw
from clustering.evaluate import bootstrap_stability, compute_internal_metrics
from clustering.models.gmm import bgmm_factory, gmm_bic_fn, gmm_factory
from clustering.preprocess import (
    PreprocessStrategy,
    build_preprocessor,
    infer_column_types,
)
from clustering.submission import write_result_json

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
random.seed(42)

# %%
df = load_raw()
types = infer_column_types(df)

STRATEGY_NAME = "quantile+freq+yeo"
strategy = PreprocessStrategy("quantile", "target_freq", True)
X = build_preprocessor(strategy, types).fit_transform(df.drop(columns=types.excluded))

# %% [markdown]
# ## Per-covariance-type BIC sweeps

# %%
K_MAX = int(os.environ.get("GMM_K_MAX", "30"))
GMM_N_INIT = int(os.environ.get("GMM_N_INIT", "3"))
STABILITY_N_ITER = int(os.environ.get("GMM_STABILITY_N_ITER", "10"))
BGMM_STABILITY_N_ITER = int(os.environ.get("BGMM_STABILITY_N_ITER", "5"))
K_RANGE = range(2, K_MAX + 1)
COV_TYPES = ["full", "tied", "diag", "spherical"]
bic_curves: dict[str, list[float]] = {}
for cov in COV_TYPES:
    bic_fn = gmm_bic_fn(covariance_type=cov, n_init=GMM_N_INIT)
    bics = [bic_fn(k, X) for k in K_RANGE]
    bic_curves[cov] = bics
    print(f"{cov}: min BIC at k={list(K_RANGE)[int(np.argmin(bics))]}")

fig, ax = plt.subplots(figsize=(8, 5))
for cov, bics in bic_curves.items():
    ax.plot(list(K_RANGE), bics, marker="o", label=cov)
ax.set_xlabel("n_components")
ax.set_ylabel("BIC")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("GMM BIC vs k by covariance type")
plt.close(fig)

# %%
best_cov_by_bic = min(
    COV_TYPES,
    key=lambda cov: min(bic_curves[cov]),
)
BEST_COV = best_cov_by_bic
BEST_K = list(K_RANGE)[int(np.argmin(bic_curves[BEST_COV]))]
print("BEST_COV:", BEST_COV)
print("BEST_K:", BEST_K)

# %% [markdown]
# ## Full signal table for the chosen (cov, k)

# %%
fp_gmm = gmm_factory(BEST_K, covariance_type=BEST_COV, n_init=max(GMM_N_INIT, 3))
labels_gmm = fp_gmm(X)
bic = gmm_bic_fn(covariance_type=BEST_COV, n_init=max(GMM_N_INIT, 3))(BEST_K, X)
m_gmm = compute_internal_metrics(X, labels_gmm, bic=bic)
stab_gmm = bootstrap_stability(fp_gmm, X, n_iter=STABILITY_N_ITER)
print("GMM at best:", m_gmm.to_dict(), stab_gmm.to_dict())

# %% [markdown]
# ## BayesianGaussianMixture for automatic k selection

# %%
fp_bgmm = bgmm_factory(n_components=min(30, max(K_RANGE)), covariance_type=BEST_COV)
labels_bgmm = fp_bgmm(X)
effective_k = int(np.unique(labels_bgmm).size)
print("BGMM effective k:", effective_k)
m_bgmm = compute_internal_metrics(X, labels_bgmm)
stab_bgmm = bootstrap_stability(fp_bgmm, X, n_iter=BGMM_STABILITY_N_ITER)
print("BGMM:", m_bgmm.to_dict(), stab_bgmm.to_dict())

# %%
Path("runs/04_gmm").mkdir(parents=True, exist_ok=True)
np.save("runs/04_gmm/labels_gmm_best.npy", labels_gmm)
np.save("runs/04_gmm/labels_bgmm.npy", labels_bgmm)

# %%
write_result_json(
    "runs/04_gmm",
    {
        "notebook": "04_gmm_bgmm",
        "preprocess_strategy": STRATEGY_NAME,
        "k_range": [min(K_RANGE), max(K_RANGE)],
        "gmm_n_init": GMM_N_INIT,
        "stability_n_iter": STABILITY_N_ITER,
        "bgmm_stability_n_iter": BGMM_STABILITY_N_ITER,
        "best_model": {
            "algorithm": "GaussianMixture",
            "params": {
                "n_components": BEST_K,
                "covariance_type": BEST_COV,
                "n_init": max(GMM_N_INIT, 3),
                "random_state": 42,
            },
            "internal_metrics": m_gmm.to_dict(),
            "stability": stab_gmm.to_dict(),
        },
        "alt_model": {
            "algorithm": "BayesianGaussianMixture",
            "params": {
                "n_components_max": min(30, max(K_RANGE)),
                "covariance_type": BEST_COV,
                "effective_k": effective_k,
            },
            "internal_metrics": m_bgmm.to_dict(),
            "stability": stab_bgmm.to_dict(),
        },
        "bic_curves": {cov: bic_curves[cov] for cov in COV_TYPES},
        "labels_path": "runs/04_gmm/labels_gmm_best.npy",
    },
)
