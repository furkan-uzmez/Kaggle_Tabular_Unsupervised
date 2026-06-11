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
# # 06 -- Agglomerative + Spectral
#
# - Agglomerative: sweep linkage in {ward, complete, average} x k.
# - Spectral: nearest_neighbors first; RBF only if n < 20k or with subsample
#   propagation otherwise.

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
import scipy.cluster.hierarchy as sch

from clustering.data import load_raw
from clustering.evaluate import bootstrap_stability, compute_internal_metrics, scan_n_clusters
from clustering.models.hierarchical import agglomerative_factory, spectral_factory
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

# %% [markdown]
# ## Agglomerative: subsample for dendrogram + memory

# %%
N_SUB = min(int(os.environ.get("AGG_N_SUB", "15000")), X.shape[0])
rng = np.random.default_rng(42)
sub_idx = rng.choice(X.shape[0], size=N_SUB, replace=False)
X_sub = X[sub_idx]

# %%
# Ward dendrogram on subsample
linkage_matrix = sch.linkage(X_sub, method="ward")
fig, ax = plt.subplots(figsize=(10, 4))
sch.dendrogram(linkage_matrix, no_labels=True, ax=ax, truncate_mode="level", p=10)
ax.set_title("Ward dendrogram (subsample)")
plt.close(fig)

# %% [markdown]
# ## Linkage x k sweep on subsample

# %%
K_MAX = int(os.environ.get("AGG_K_MAX", "20"))
STABILITY_N_ITER = int(os.environ.get("AGG_STABILITY_N_ITER", "5"))
K_RANGE = range(2, K_MAX + 1)
LINKAGES = ["ward", "complete", "average"]
agg_results = {}
for linkage in LINKAGES:
    table = scan_n_clusters(
        lambda k, lk=linkage: agglomerative_factory(k, linkage=lk),
        X_sub,
        k_range=K_RANGE,
        stability_n_iter=STABILITY_N_ITER,
        stability_sample_frac=0.7,
    )
    agg_results[linkage] = table
    fig = plot_signal_curves(table, title=f"Agglomerative ({linkage}) on subsample")
    plt.close(fig)

# %%
best_linkage, best_k, best_silhouette = max(
    (
        (linkage, k, entry["metrics"].silhouette)
        for linkage, table in agg_results.items()
        for k, entry in table.items()
        if entry["metrics"].silhouette is not None
    ),
    key=lambda item: item[2],
)
BEST_LINKAGE = best_linkage
BEST_K = int(best_k)
print("BEST_LINKAGE:", BEST_LINKAGE)
print("BEST_K:", BEST_K)
print("best silhouette:", best_silhouette)

USE_FULL_DATA_FOR_AGG = os.environ.get("AGG_USE_FULL_DATA", "0") == "1"
X_for_agg = X if USE_FULL_DATA_FOR_AGG else X_sub
fp_agg = agglomerative_factory(BEST_K, linkage=BEST_LINKAGE)
labels_agg = fp_agg(X_for_agg)
m_agg = compute_internal_metrics(X_for_agg, labels_agg)
stab_agg = bootstrap_stability(fp_agg, X_for_agg, n_iter=STABILITY_N_ITER)
print("Agglomerative best:", m_agg.to_dict(), stab_agg.to_dict())

# %% [markdown]
# ## Spectral: nearest_neighbors affinity

# %%
SPECTRAL_N_SUB = min(int(os.environ.get("SPECTRAL_N_SUB", "5000")), X.shape[0])
spectral_idx = rng.choice(X.shape[0], size=SPECTRAL_N_SUB, replace=False)
X_spec = X[spectral_idx]
SPECTRAL_STABILITY_N_ITER = int(os.environ.get("SPECTRAL_STABILITY_N_ITER", "3"))
fp_spec = spectral_factory(BEST_K, affinity="nearest_neighbors", n_neighbors=10)
labels_spec = fp_spec(X_spec)
m_spec = compute_internal_metrics(X_spec, labels_spec)
stab_spec = bootstrap_stability(fp_spec, X_spec, n_iter=SPECTRAL_STABILITY_N_ITER)
print("Spectral knn:", m_spec.to_dict(), stab_spec.to_dict())

# %%
Path("runs/06_hier_spectral").mkdir(parents=True, exist_ok=True)
if USE_FULL_DATA_FOR_AGG:
    labels_for_consensus = labels_agg
    labels_path = "runs/06_hier_spectral/labels_agglomerative.npy"
else:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=5).fit(X_sub)
    _, idx_nbr = nn.kneighbors(X)
    nbr_labels = labels_agg[idx_nbr]
    labels_for_consensus = np.empty(X.shape[0], dtype=int)
    for i, row in enumerate(nbr_labels):
        vals, counts = np.unique(row, return_counts=True)
        labels_for_consensus[i] = int(vals[np.argmax(counts)])
    labels_path = "runs/06_hier_spectral/labels_agglomerative.npy"

np.save(labels_path, labels_for_consensus)
np.save("runs/06_hier_spectral/labels_spectral.npy", labels_spec)

# %%
write_result_json(
    "runs/06_hier_spectral",
    {
        "notebook": "06_agglomerative_spectral",
        "preprocess_strategy": STRATEGY_NAME,
        "subsample_size": N_SUB,
        "k_range": [min(K_RANGE), max(K_RANGE)],
        "agglomerative_stability_n_iter": STABILITY_N_ITER,
        "used_full_data_for_agg": USE_FULL_DATA_FOR_AGG,
        "spectral_subsample_size": SPECTRAL_N_SUB,
        "spectral_stability_n_iter": SPECTRAL_STABILITY_N_ITER,
        "best_model": {
            "algorithm": "AgglomerativeClustering",
            "params": {"n_clusters": BEST_K, "linkage": BEST_LINKAGE},
            "internal_metrics": m_agg.to_dict(),
            "stability": stab_agg.to_dict(),
        },
        "alt_model": {
            "algorithm": "SpectralClustering",
            "params": {"n_clusters": BEST_K, "affinity": "nearest_neighbors", "n_neighbors": 10},
            "internal_metrics": m_spec.to_dict(),
            "stability": stab_spec.to_dict(),
        },
        "labels_path": labels_path,
    },
)
