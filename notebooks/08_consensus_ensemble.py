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
# # 08 -- Consensus clustering
#
# Load the best label set from each of 03-07, build the co-association matrix,
# and cluster on `1 - M` with average linkage.

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
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from clustering.data import load_raw
from clustering.dim_reduction import fit_pca
from clustering.evaluate import compute_internal_metrics
from clustering.models.consensus import co_association_matrix, consensus_labels
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

# Use a neutral preprocess for downstream scoring/visualization only.
strategy = PreprocessStrategy("quantile", "ordinal", False)
X = build_preprocessor(strategy, types).fit_transform(df.drop(columns=types.excluded))

# %% [markdown]
# ## Collect label sets

# %%
LABEL_PATHS = {
    "kmeans": "runs/03_kmeans/labels_best.npy",
    "gmm": "runs/04_gmm/labels_gmm_best.npy",
    "hdbscan": "runs/05_density/labels_hdbscan.npy",
    "agglomerative": "runs/06_hier_spectral/labels_agglomerative.npy",
    "dim_red": "runs/07_dim_reduction/labels_best.npy",
}

label_sets: dict[str, np.ndarray] = {}
for name, path in LABEL_PATHS.items():
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run notebook {name} first.")
    arr = np.load(p)
    if arr.shape[0] != X.shape[0]:
        raise ValueError(f"{name}: expected {X.shape[0]} rows, got {arr.shape[0]}")
    label_sets[name] = arr
    print(f"{name}: {arr.shape}, {len(np.unique(arr))} unique labels")

# %% [markdown]
# ## Build co-association matrix and consensus labels
#
# This allocates an `N x N` float32 matrix, so it uses a bounded subsample and
# propagates the consensus labels back to the full dataset via 5-NN.

# %%
N_FULL = X.shape[0]
COASSOC_MAX_N = min(int(os.environ.get("COASSOC_MAX_N", "20000")), N_FULL)
if N_FULL > COASSOC_MAX_N:
    print(f"N={N_FULL} > {COASSOC_MAX_N}; computing consensus on a subsample.")
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(N_FULL, size=COASSOC_MAX_N, replace=False)
    label_sets_for_M = [arr[sub_idx] for arr in label_sets.values()]
    X_for_scoring = X[sub_idx]
else:
    sub_idx = None
    label_sets_for_M = list(label_sets.values())
    X_for_scoring = X

# %%
unique_counts = [len(np.unique(arr)) for arr in label_sets_for_M]
CONSENSUS_K = int(os.environ.get("CONSENSUS_K", str(int(round(np.median(unique_counts))))))

M = co_association_matrix(label_sets_for_M)
print("M shape:", M.shape, "diag mean:", float(np.diag(M).mean()))

consensus = consensus_labels(label_sets_for_M, n_clusters=CONSENSUS_K, linkage="average")
print("consensus unique:", sorted(set(consensus.tolist())))

# %% [markdown]
# ## Score consensus

# %%
m_cons = compute_internal_metrics(X_for_scoring, consensus)

loo_aris: list[float] = []
for i in range(len(label_sets_for_M)):
    reduced = label_sets_for_M[:i] + label_sets_for_M[i + 1 :]
    cl = consensus_labels(reduced, n_clusters=CONSENSUS_K, linkage="average")
    loo_aris.append(float(adjusted_rand_score(consensus, cl)))
print("Leave-one-out ARI mean:", float(np.mean(loo_aris)), "std:", float(np.std(loo_aris)))

# %% [markdown]
# ## Visualize on PCA(2)

# %%
Z, _ = fit_pca(X_for_scoring, 2)
fig = plot_2d_clusters(Z, consensus, title=f"Consensus (k={CONSENSUS_K})")
plt.close(fig)

# %% [markdown]
# ## Save: propagate consensus labels to the full dataset (if subsampled)

# %%
if sub_idx is not None:
    nn = NearestNeighbors(n_neighbors=5).fit(X[sub_idx])
    _, idx_nbr = nn.kneighbors(X)
    nbr_labels = consensus[idx_nbr]
    full_labels = np.empty(N_FULL, dtype=int)
    for i, row in enumerate(nbr_labels):
        vals, counts = np.unique(row, return_counts=True)
        full_labels[i] = int(vals[np.argmax(counts)])
else:
    full_labels = consensus

Path("runs/08_consensus").mkdir(parents=True, exist_ok=True)
np.save("runs/08_consensus/labels_consensus.npy", full_labels)

# %%
write_result_json(
    "runs/08_consensus",
    {
        "notebook": "08_consensus_ensemble",
        "input_label_sets": LABEL_PATHS,
        "consensus_n_clusters": CONSENSUS_K,
        "consensus_linkage": "average",
        "subsampled_to": COASSOC_MAX_N if sub_idx is not None else None,
        "best_model": {
            "algorithm": "Consensus (average linkage on co-assoc)",
            "params": {
                "n_clusters": CONSENSUS_K,
                "n_label_sets": len(label_sets_for_M),
            },
            "internal_metrics": m_cons.to_dict(),
            "stability": {
                "method": "leave-one-out ARI on consensus",
                "mean_ari": float(np.mean(loo_aris)),
                "std_ari": float(np.std(loo_aris)),
                "per_iter": loo_aris,
            },
        },
        "labels_path": "runs/08_consensus/labels_consensus.npy",
    },
)
