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
# # 05 -- DBSCAN + HDBSCAN
#
# - DBSCAN: pick eps via k-distance "knee".
# - HDBSCAN: sweep min_cluster_size.
# - Decide noise-point treatment: nearest-cluster assignment vs. keep as own id.

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
import pandas as pd

from clustering.data import load_raw
from clustering.evaluate import bootstrap_stability, compute_internal_metrics
from clustering.models.density import (
    assign_noise_to_nearest_cluster,
    dbscan_factory,
    hdbscan_factory,
    k_distance_curve,
)
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

STRATEGY_NAME = "quantile+ordinal"
strategy = PreprocessStrategy("quantile", "ordinal", False)
X = build_preprocessor(strategy, types).fit_transform(df.drop(columns=types.excluded))

# %% [markdown]
# ## DBSCAN: k-distance graph for eps

# %%
K_FOR_KDIST = 5  # rule of thumb: min_samples
curve = k_distance_curve(X, k=K_FOR_KDIST)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve)
ax.set_xlabel("points sorted")
ax.set_ylabel(f"distance to {K_FOR_KDIST}-th neighbor")
ax.set_title("k-distance graph (pick eps at the knee)")
ax.grid(True, alpha=0.3)
plt.close(fig)

# %%
EPS = float(os.environ.get("DBSCAN_EPS", str(float(np.quantile(curve, 0.95)))))
MIN_SAMPLES = 5

fp_db = dbscan_factory(eps=EPS, min_samples=MIN_SAMPLES)
labels_db = fp_db(X)
n_noise_db = int((labels_db == -1).sum())
n_clusters_db = int(np.unique(labels_db[labels_db != -1]).size)
print(f"DBSCAN: {n_clusters_db} clusters, {n_noise_db} noise ({n_noise_db / len(X):.1%})")

m_db = compute_internal_metrics(X, labels_db, exclude_noise=True)
print("DBSCAN metrics (noise excluded):", m_db.to_dict())

# %% [markdown]
# ## HDBSCAN: sweep min_cluster_size

# %%
MCS_GRID = [int(v) for v in os.environ.get("HDBSCAN_MCS_GRID", "30,50,100,200,400").split(",")]
hdb_results = []
for mcs in MCS_GRID:
    fp = hdbscan_factory(min_cluster_size=mcs)
    labels = fp(X)
    n_noise = int((labels == -1).sum())
    n_clust = int(np.unique(labels[labels != -1]).size)
    m = compute_internal_metrics(X, labels, exclude_noise=True)
    hdb_results.append(
        {
            "min_cluster_size": mcs,
            "n_clusters": n_clust,
            "n_noise": n_noise,
            "noise_frac": n_noise / len(X),
            "silhouette": m.silhouette,
            "davies_bouldin": m.davies_bouldin,
            "calinski_harabasz": m.calinski_harabasz,
        }
    )
pd.DataFrame(hdb_results)

# %%
valid_hdb = [row for row in hdb_results if row["silhouette"] is not None]
BEST_MCS = max(valid_hdb, key=lambda row: row["silhouette"])["min_cluster_size"] if valid_hdb else MCS_GRID[0]
fp_hdb = hdbscan_factory(min_cluster_size=BEST_MCS)
labels_hdb = fp_hdb(X)
n_noise_hdb = int((labels_hdb == -1).sum())
print(f"HDBSCAN @ mcs={BEST_MCS}: noise = {n_noise_hdb} ({n_noise_hdb / len(X):.1%})")

# %% [markdown]
# ## Noise treatment decision
#
# Use nearest-cluster assignment for the baseline because later consensus and
# submission code expect a cluster label for every row. The raw HDBSCAN noise
# fraction remains recorded for interpretability.

# %%
NOISE_POLICY = os.environ.get("HDBSCAN_NOISE_POLICY", "nearest")

if NOISE_POLICY == "nearest":
    labels_hdb_final = assign_noise_to_nearest_cluster(X, labels_hdb, k=5)
elif NOISE_POLICY == "own_cluster":
    labels_hdb_final = labels_hdb.copy()
    noise_mask = labels_hdb_final == -1
    labels_hdb_final[noise_mask] = labels_hdb_final.max() + 1
else:
    raise ValueError(f"Unknown NOISE_POLICY: {NOISE_POLICY}")

m_hdb = compute_internal_metrics(X, labels_hdb_final, exclude_noise=False)
HDBSCAN_STABILITY_N_ITER = int(os.environ.get("HDBSCAN_STABILITY_N_ITER", "5"))
stab_hdb = bootstrap_stability(fp_hdb, X, n_iter=HDBSCAN_STABILITY_N_ITER)
print("HDBSCAN final:", m_hdb.to_dict(), stab_hdb.to_dict())

# %%
Path("runs/05_density").mkdir(parents=True, exist_ok=True)
np.save("runs/05_density/labels_hdbscan.npy", labels_hdb_final)

# %%
write_result_json(
    "runs/05_density",
    {
        "notebook": "05_density_based",
        "preprocess_strategy": STRATEGY_NAME,
        "hdbscan_stability_n_iter": HDBSCAN_STABILITY_N_ITER,
        "best_model": {
            "algorithm": "HDBSCAN",
            "params": {
                "min_cluster_size": BEST_MCS,
                "noise_policy": NOISE_POLICY,
            },
            "internal_metrics": m_hdb.to_dict(),
            "stability": stab_hdb.to_dict(),
        },
        "alt_model": {
            "algorithm": "DBSCAN",
            "params": {"eps": EPS, "min_samples": MIN_SAMPLES},
            "n_clusters": n_clusters_db,
            "n_noise": n_noise_db,
            "internal_metrics": m_db.to_dict(),
        },
        "hdbscan_sweep": hdb_results,
        "labels_path": "runs/05_density/labels_hdbscan.npy",
    },
)
