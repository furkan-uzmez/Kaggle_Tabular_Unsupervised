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
# # 02 -- Preprocess strategy experiments
#
# Compare 6 preprocessing strategies. For each:
#   1. Build the pipeline, fit_transform the data.
#   2. Quick KMeans(k=10) silhouette as a rough quality signal.
#   3. PCA(2) plot colored by KMeans labels.
#
# Pick 2-3 winners to use in later notebooks. Also identify the
# "GMM-friendly" strategy (likely standard+yeo-johnson).

# %%
import random
import warnings
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os

    os.chdir("..")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering.data import load_raw
from clustering.dim_reduction import fit_pca
from clustering.evaluate import compute_internal_metrics
from clustering.models.kmeans import kmeans_factory
from clustering.preprocess import (
    PreprocessStrategy,
    build_preprocessor,
    infer_column_types,
)
from clustering.submission import write_result_json
from clustering.viz import plot_2d_clusters

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
random.seed(42)

# %%
df = load_raw()
types = infer_column_types(df)
X_input = df.drop(columns=types.excluded)
print("input shape:", X_input.shape)

# %%
STRATEGIES: dict[str, PreprocessStrategy] = {
    "standard+onehot": PreprocessStrategy("standard", "onehot", False),
    "standard+onehot+yeo": PreprocessStrategy("standard", "onehot", True),
    "robust+ordinal": PreprocessStrategy("robust", "ordinal", False),
    "robust+freq": PreprocessStrategy("robust", "target_freq", False),
    "quantile+ordinal": PreprocessStrategy("quantile", "ordinal", False),
    "quantile+freq+yeo": PreprocessStrategy("quantile", "target_freq", True),
}

K_PROBE = 10
results: dict[str, dict] = {}
for name, strategy in STRATEGIES.items():
    print(f"--- {name} ---")
    X = build_preprocessor(strategy, types).fit_transform(X_input)
    print("  shape:", X.shape)
    labels = kmeans_factory(K_PROBE, n_init=10)(X)
    metrics = compute_internal_metrics(X, labels)
    print(
        f"  silhouette={metrics.silhouette:.3f}  "
        f"DB={metrics.davies_bouldin:.3f}  CH={metrics.calinski_harabasz:.1f}"
    )
    Z, _ = fit_pca(X, 2)
    fig = plot_2d_clusters(Z, labels, title=f"{name} -- KMeans(k={K_PROBE}) on PCA(2)")
    plt.close(fig)
    results[name] = {
        "shape": list(X.shape),
        "silhouette": metrics.silhouette,
        "davies_bouldin": metrics.davies_bouldin,
        "calinski_harabasz": metrics.calinski_harabasz,
    }

# %%
# Summary table
summary = pd.DataFrame(results).T.sort_values("silhouette", ascending=False)
summary

# %%
best_general = summary.index[0]
best_db = summary["davies_bouldin"].astype(float).idxmin()
best_ch = summary["calinski_harabasz"].astype(float).idxmax()
gmm_friendly = "quantile+freq+yeo" if "quantile+freq+yeo" in STRATEGIES else best_general
print("best_general:", best_general)
print("best_db:", best_db)
print("best_ch:", best_ch)
print("gmm_friendly:", gmm_friendly)

# %% [markdown]
# ## Winners
#
# - General-purpose winner(s): `quantile+ordinal`; it had the best silhouette (0.0444), lowest Davies-Bouldin (3.2404), and highest Calinski-Harabasz (2840.4) in the KMeans(k=10) probe.
# - GMM-friendly (Gaussian-shaped after transform): `quantile+freq+yeo`, because quantile scaling plus Yeo-Johnson is the most Gaussianizing option in the planned strategy set, even though it was not the KMeans probe winner.
# - Rejected and why: `standard+onehot`, `standard+onehot+yeo`, `robust+ordinal`, and `robust+freq` had weaker probe metrics; they are deprioritized unless a later model family needs them for a specific reason.

# %%
write_result_json(
    "runs/02_preprocess",
    {
        "notebook": "02_preprocess_experiments",
        "k_probe": K_PROBE,
        "best_general_strategy": best_general,
        "best_davies_bouldin_strategy": best_db,
        "best_calinski_harabasz_strategy": best_ch,
        "gmm_friendly_strategy": gmm_friendly,
        "strategies": {
            name: {
                "config": {
                    "continuous_scaler": s.continuous_scaler,
                    "categorical_encoder": s.categorical_encoder,
                    "power_transform": s.power_transform,
                },
                "result": results[name],
            }
            for name, s in STRATEGIES.items()
        },
    },
)
