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
# # 01 -- Exploratory Data Analysis
#
# Goals: confirm row count, dtypes, missing values, column-type heuristic,
# basic distributions, and a raw 2D snapshot (PCA + UMAP without preprocessing
# decisions baked in).

# %%
import random
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clustering.data import load_raw, load_sample_submission
from clustering.dim_reduction import fit_pca, fit_umap
from clustering.preprocess import (
    PreprocessStrategy,
    build_preprocessor,
    infer_column_types,
)
from clustering.submission import write_result_json

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
random.seed(42)

if Path.cwd().name == "notebooks":
    import os

    os.chdir("..")

# %% [markdown]
# ## 1. Load

# %%
df = load_raw()
sub = load_sample_submission()
print("data shape:", df.shape)
print("sample_submission shape:", sub.shape)
df.head()

# %% [markdown]
# ## 2. Schema & missing values

# %%
df.dtypes

# %%
missing = df.isna().sum()
print("Columns with missing values:")
print(missing[missing > 0] if (missing > 0).any() else "(none)")

# %% [markdown]
# ## 3. Column-type heuristic

# %%
types = infer_column_types(df)
print(f"continuous ({len(types.continuous)}):", types.continuous)
print(f"categorical ({len(types.categorical)}):", types.categorical)
print("excluded:", types.excluded)

# %%
# Per-column nunique to sanity-check the heuristic
nunique_summary = pd.DataFrame(
    {
        "dtype": df.dtypes,
        "nunique": df.nunique(),
        "bucket": [
            "excluded"
            if c in types.excluded
            else ("continuous" if c in types.continuous else "categorical")
            for c in df.columns
        ],
    }
)
nunique_summary.sort_values(["bucket", "nunique"])

# %% [markdown]
# ## 4. Continuous-column distributions

# %%
n = len(types.continuous)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
for ax, col in zip(axes.flatten(), types.continuous):
    sns.histplot(df[col], bins=50, ax=ax, kde=True)
    ax.set_title(col, fontsize=9)
for ax in axes.flatten()[n:]:
    ax.axis("off")
fig.tight_layout()
plt.close(fig)

# %%
df[types.continuous].describe().T

# %% [markdown]
# ## 5. Categorical-column value counts

# %%
for col in types.categorical:
    print(col, "->", df[col].value_counts(dropna=False).to_dict())

# %% [markdown]
# ## 6. Correlation heatmap (continuous only)

# %%
max_abs_corr = None
strong_corr_pairs = 0
if len(types.continuous) >= 2:
    corr = df[types.continuous].corr()
    corr_values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    max_abs_corr = float(corr_values.abs().max()) if not corr_values.empty else None
    strong_corr_pairs = int((corr_values.abs() > 0.7).sum())
    fig, ax = plt.subplots(
        figsize=(min(0.5 * len(corr) + 2, 12), min(0.5 * len(corr) + 2, 12))
    )
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax)
    plt.close(fig)

# %% [markdown]
# ## 7. Raw 2D snapshot (PCA + UMAP)
#
# Use the minimum-decision preprocessor (standard scaling, ordinal encoding,
# no power transform) so the snapshot reflects the data, not a strategy choice.

# %%
baseline = PreprocessStrategy(
    continuous_scaler="standard", categorical_encoder="ordinal", power_transform=False
)
X = build_preprocessor(baseline, types).fit_transform(df.drop(columns=types.excluded))
print("X shape after baseline preprocess:", X.shape)

# %%
Z_pca, pca_model = fit_pca(X, n_components=2)
pca_explained = pca_model.explained_variance_ratio_.tolist()
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(Z_pca[:, 0], Z_pca[:, 1], s=3, alpha=0.4)
ax.set_title("PCA(2) raw snapshot")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.close(fig)

# %%
# UMAP on a subsample to keep this notebook snappy
sample_size = min(15000, X.shape[0])
rng = np.random.default_rng(42)
idx = rng.choice(X.shape[0], size=sample_size, replace=False)
Z_umap, _ = fit_umap(X[idx], n_components=2, n_neighbors=30, min_dist=0.1)
umap_spread = {
    "dim0_min": float(Z_umap[:, 0].min()),
    "dim0_max": float(Z_umap[:, 0].max()),
    "dim1_min": float(Z_umap[:, 1].min()),
    "dim1_max": float(Z_umap[:, 1].max()),
}
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(Z_umap[:, 0], Z_umap[:, 1], s=3, alpha=0.4)
ax.set_title(f"UMAP(2) raw snapshot (n={sample_size})")
plt.close(fig)

# %%
write_result_json(
    "runs/01_eda",
    {
        "notebook": "01_eda",
        "data_shape": list(df.shape),
        "sample_submission_shape": list(sub.shape),
        "missing_total": int(missing.sum()),
        "missing_columns": missing[missing > 0].to_dict(),
        "continuous_columns": types.continuous,
        "categorical_columns": types.categorical,
        "excluded_columns": types.excluded,
        "max_abs_continuous_corr": max_abs_corr,
        "strong_corr_pairs_abs_gt_0_7": strong_corr_pairs,
        "baseline_preprocessed_shape": list(X.shape),
        "pca_2_explained_variance_ratio": pca_explained,
        "umap_sample_size": sample_size,
        "umap_spread": umap_spread,
    },
)

# %% [markdown]
# ## 8. EDA findings
#
# - Row count: 98,000 rows and 30 columns in `data.csv`; `sample_submission.csv` has 98,000 rows and 2 columns.
# - Missing values: none observed in `data.csv` during the executed run.
# - Continuous columns: 29 feature columns (`f_00` through `f_28`) were inferred as continuous.
# - Categorical columns: none were inferred with the current low-cardinality heuristic on this dataset.
# - Excluded columns: the raw file uses lowercase `id`, while the sample submission uses uppercase `Id`; preprocessing excludes the raw `id` identifier.
# - Correlation structure: no continuous feature pairs exceeded |r| > 0.7; maximum absolute pairwise correlation was recorded in `runs/01_eda/result.json`.
# - Raw PCA snapshot: PCA(2) explains only a minority of variance, so it is useful as a coarse sanity view rather than a decisive clustering basis.
# - Raw UMAP snapshot: UMAP was run on a 15,000-row deterministic subsample for a qualitative manifold snapshot.
# - Decision notes for `02_preprocess_experiments`: compare scaling/encoding strategies systematically; keep `Id` excluded, test frequency encoding for low-cardinality columns, and do not choose downstream models from this visual EDA alone.
