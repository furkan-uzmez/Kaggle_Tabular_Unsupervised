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
# # 07 -- Dim reduction + re-clustering
#
# - PCA: pick n_components by 95% cumulative variance.
# - UMAP: try (n_components, n_neighbors) combinations.
# - Re-run the 2 strongest algorithms from 03-06 on each embedding.

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
from clustering.dim_reduction import fit_pca, fit_umap, pca_variance_curve
from clustering.evaluate import bootstrap_stability, compute_internal_metrics
from clustering.models.gmm import gmm_factory
from clustering.models.kmeans import kmeans_factory
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
X_full = build_preprocessor(strategy, types).fit_transform(df.drop(columns=types.excluded))

DIMRED_N_SUB = min(int(os.environ.get("DIMRED_N_SUB", str(X_full.shape[0]))), X_full.shape[0])
rng = np.random.default_rng(42)
sub_idx = rng.choice(X_full.shape[0], size=DIMRED_N_SUB, replace=False)
X = X_full[sub_idx]

# %% [markdown]
# ## PCA component selection

# %%
variance = pca_variance_curve(X)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(variance) + 1), variance, marker="o")
ax.axhline(0.95, color="r", linestyle="--", label="95%")
ax.set_xlabel("n_components")
ax.set_ylabel("cumulative explained variance")
ax.legend()
ax.grid(True, alpha=0.3)
plt.close(fig)

PCA_K = int(np.searchsorted(variance, 0.95) + 1)
print("PCA components for 95% variance:", PCA_K)

# %%
Z_pca, _ = fit_pca(X, n_components=PCA_K)

# %% [markdown]
# ## UMAP grid

# %%
UMAP_GRID = [
    {"n_components": 2, "n_neighbors": 15},
    {"n_components": 2, "n_neighbors": 30},
    {"n_components": 5, "n_neighbors": 30},
    {"n_components": 10, "n_neighbors": 30},
]
UMAP_LIMIT = int(os.environ.get("UMAP_GRID_LIMIT", str(len(UMAP_GRID))))
UMAP_GRID = UMAP_GRID[:UMAP_LIMIT]
umap_embeddings: dict[str, np.ndarray] = {}
for cfg in UMAP_GRID:
    name = f"umap_d{cfg['n_components']}_nn{cfg['n_neighbors']}"
    print(f"fitting {name} ...")
    Z, _ = fit_umap(X, **cfg)
    umap_embeddings[name] = Z
    print(f"  shape: {Z.shape}")

# %% [markdown]
# ## Re-cluster on each embedding with KMeans and GMM

# %%
TARGET_K = int(os.environ.get("DIMRED_TARGET_K", "3"))
ALGORITHMS = {
    "KMeans": lambda Z, k=TARGET_K: kmeans_factory(k, n_init=10)(Z),
    "GMM_full": lambda Z, k=TARGET_K: gmm_factory(k, covariance_type="full", n_init=3)(Z),
}

embeddings = {"pca": Z_pca, **umap_embeddings}
rows = []
labels_store: dict[tuple[str, str], np.ndarray] = {}
for emb_name, Z in embeddings.items():
    for algo_name, algo in ALGORITHMS.items():
        labels = algo(Z)
        m = compute_internal_metrics(Z, labels)
        rows.append(
            {
                "embedding": emb_name,
                "algorithm": algo_name,
                "silhouette": m.silhouette,
                "davies_bouldin": m.davies_bouldin,
                "calinski_harabasz": m.calinski_harabasz,
            }
        )
        labels_store[(emb_name, algo_name)] = labels
pd.DataFrame(rows).sort_values("silhouette", ascending=False)

# %%
best_row = max(rows, key=lambda row: row["silhouette"] if row["silhouette"] is not None else -np.inf)
BEST_EMB = best_row["embedding"]
BEST_ALGO = best_row["algorithm"]
best_labels_sub = labels_store[(BEST_EMB, BEST_ALGO)]
best_metrics = compute_internal_metrics(embeddings[BEST_EMB], best_labels_sub)
fp_best = lambda Z: ALGORITHMS[BEST_ALGO](Z)
DIMRED_STABILITY_N_ITER = int(os.environ.get("DIMRED_STABILITY_N_ITER", "5"))
best_stab = bootstrap_stability(fp_best, embeddings[BEST_EMB], n_iter=DIMRED_STABILITY_N_ITER)

# %%
if DIMRED_N_SUB < X_full.shape[0]:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=5).fit(X_full[sub_idx])
    _, idx_nbr = nn.kneighbors(X_full)
    nbr_labels = best_labels_sub[idx_nbr]
    best_labels = np.empty(X_full.shape[0], dtype=int)
    for i, row in enumerate(nbr_labels):
        vals, counts = np.unique(row, return_counts=True)
        best_labels[i] = int(vals[np.argmax(counts)])
else:
    best_labels = best_labels_sub

Path("runs/07_dim_reduction").mkdir(parents=True, exist_ok=True)
np.save("runs/07_dim_reduction/labels_best.npy", best_labels)

# %%
write_result_json(
    "runs/07_dim_reduction",
    {
        "notebook": "07_dim_reduction_clustering",
        "preprocess_strategy": STRATEGY_NAME,
        "subsample_size": DIMRED_N_SUB,
        "umap_grid_limit": UMAP_LIMIT,
        "pca_n_components": PCA_K,
        "target_k": TARGET_K,
        "stability_n_iter": DIMRED_STABILITY_N_ITER,
        "best_model": {
            "algorithm": f"{BEST_ALGO} on {BEST_EMB}",
            "params": {"n_clusters": TARGET_K},
            "internal_metrics": best_metrics.to_dict(),
            "stability": best_stab.to_dict(),
        },
        "comparison_table": rows,
        "labels_path": "runs/07_dim_reduction/labels_best.npy",
    },
)
