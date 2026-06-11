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
# # 03 -- KMeans / MiniBatchKMeans baseline
#
# Sweep n_clusters in [2, 30] with 4 internal signals + bootstrap stability.
# Use the general-purpose winning preprocess strategy from notebook 02.

# %%
import random
import warnings
import os
from pathlib import Path

if Path.cwd().name == "notebooks":
    os.chdir("..")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering.data import load_raw
from clustering.evaluate import (
    bootstrap_stability,
    compute_internal_metrics,
    scan_n_clusters,
)
from clustering.models.kmeans import kmeans_factory, minibatch_kmeans_factory
from clustering.preprocess import (
    PreprocessStrategy,
    build_preprocessor,
    infer_column_types,
)
from clustering.submission import write_result_json
from clustering.viz import plot_signal_curves

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
random.seed(42)

# %%
df = load_raw()
types = infer_column_types(df)

STRATEGY_NAME = "quantile+ordinal"
strategy = PreprocessStrategy("quantile", "ordinal", False)

X = build_preprocessor(strategy, types).fit_transform(df.drop(columns=types.excluded))
print("X shape:", X.shape)

# %% [markdown]
# ## Sweep n_clusters in [2, 30]

# %%
K_MAX = int(os.environ.get("KMEANS_K_MAX", "30"))
STABILITY_N_ITER = int(os.environ.get("KMEANS_STABILITY_N_ITER", "10"))
K_RANGE = range(2, K_MAX + 1)
table = scan_n_clusters(
    lambda k: kmeans_factory(k, n_init=20),
    X,
    k_range=K_RANGE,
    stability_n_iter=STABILITY_N_ITER,
    stability_sample_frac=0.8,
)
fig = plot_signal_curves(table, title=f"KMeans scan ({STRATEGY_NAME})")
plt.close(fig)

# %%
# Tabular summary
rows = []
for k, entry in table.items():
    m = entry["metrics"]
    s = entry["stability"]
    rows.append(
        {
            "k": k,
            "silhouette": m.silhouette,
            "davies_bouldin": m.davies_bouldin,
            "calinski_harabasz": m.calinski_harabasz,
            "stability_mean": s.mean_ari,
            "stability_std": s.std_ari,
        }
    )
scan_df = pd.DataFrame(rows).set_index("k")
scan_df

# %%
Path("runs/03_kmeans").mkdir(parents=True, exist_ok=True)
scan_df.to_csv("runs/03_kmeans/scan_table.csv")

# %% [markdown]
# ## Best-k decision
#
# Select by the strongest silhouette as a first-pass unsupervised baseline, then
# keep the full scan table for comparing metric disagreement and stability in
# final selection.

# %%
BEST_K = int(scan_df["silhouette"].idxmax())
print("BEST_K by silhouette:", BEST_K)

# %%
# MiniBatchKMeans comparison at the same k
fp_km = kmeans_factory(BEST_K, n_init=20)
labels_km = fp_km(X)
m_km = compute_internal_metrics(X, labels_km)
stab_km = bootstrap_stability(fp_km, X, n_iter=STABILITY_N_ITER)

fp_mb = minibatch_kmeans_factory(BEST_K, batch_size=4096, n_init=10)
labels_mb = fp_mb(X)
m_mb = compute_internal_metrics(X, labels_mb)
stab_mb = bootstrap_stability(fp_mb, X, n_iter=STABILITY_N_ITER)

pd.DataFrame(
    {
        "KMeans": {
            "silhouette": m_km.silhouette,
            "davies_bouldin": m_km.davies_bouldin,
            "calinski_harabasz": m_km.calinski_harabasz,
            "stability_mean": stab_km.mean_ari,
            "stability_std": stab_km.std_ari,
        },
        "MiniBatchKMeans": {
            "silhouette": m_mb.silhouette,
            "davies_bouldin": m_mb.davies_bouldin,
            "calinski_harabasz": m_mb.calinski_harabasz,
            "stability_mean": stab_mb.mean_ari,
            "stability_std": stab_mb.std_ari,
        },
    }
)

# %%
# Save labels for downstream consensus (08)
np.save("runs/03_kmeans/labels_best.npy", labels_km)

# %%
write_result_json(
    "runs/03_kmeans",
    {
        "notebook": "03_baseline_kmeans",
        "preprocess_strategy": STRATEGY_NAME,
        "k_range": [min(K_RANGE), max(K_RANGE)],
        "stability_n_iter": STABILITY_N_ITER,
        "best_model": {
            "algorithm": "KMeans",
            "params": {"n_clusters": BEST_K, "n_init": 20, "random_state": 42},
            "internal_metrics": m_km.to_dict(),
            "stability": stab_km.to_dict(),
        },
        "minibatch_compare": {
            "internal_metrics": m_mb.to_dict(),
            "stability": stab_mb.to_dict(),
        },
        "scan_summary_path": "runs/03_kmeans/scan_table.csv",
        "labels_path": "runs/03_kmeans/labels_best.npy",
    },
)
