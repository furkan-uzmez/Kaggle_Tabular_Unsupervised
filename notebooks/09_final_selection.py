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
# # 09 -- Final selection + submissions
#
# Aggregate all `runs/*/result.json` files into one master comparison table,
# pick 1 individual + 1 consensus candidate, write submission CSVs.

# %%
import json
import random
import warnings
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os

    os.chdir("..")

import numpy as np
import pandas as pd

from clustering.data import load_raw, load_sample_submission
from clustering.submission import write_result_json, write_submission

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ## Aggregate result.json files

# %%
RUNS_DIR = Path("runs")
rows = []
for result_path in sorted(RUNS_DIR.glob("*/result.json")):
    with result_path.open() as f:
        payload = json.load(f)
    best = payload.get("best_model", {})
    metrics = best.get("internal_metrics", {}) or {}
    stability = best.get("stability", {}) or {}
    rows.append(
        {
            "notebook": payload.get("notebook", result_path.parent.name),
            "algorithm": best.get("algorithm"),
            "preprocess": payload.get("preprocess_strategy"),
            "silhouette": metrics.get("silhouette"),
            "davies_bouldin": metrics.get("davies_bouldin"),
            "calinski_harabasz": metrics.get("calinski_harabasz"),
            "bic": metrics.get("bic"),
            "stab_mean": stability.get("mean_ari"),
            "stab_std": stability.get("std_ari"),
            "labels_path": payload.get("labels_path"),
        }
    )

master = pd.DataFrame(rows)
master

# %% [markdown]
# ## Combined-signal ranking
#
# For models with no silhouette (degenerate), they drop out. We z-score the
# four signals across rows (BIC excluded -- only meaningful within GMM), then
# average into a single combined score. Higher = better.

# %%
def zscore(s: pd.Series, *, higher_is_better: bool) -> pd.Series:
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    std = valid.std(ddof=0)
    if std == 0:
        z = pd.Series(0.0, index=s.index).where(s.notna())
    else:
        z = (s - valid.mean()) / std
    return z if higher_is_better else -z


score_cols = {
    "silhouette": True,
    "davies_bouldin": False,
    "calinski_harabasz": True,
    "stab_mean": True,
}
z_frame = pd.DataFrame(
    {col: zscore(master[col], higher_is_better=hib) for col, hib in score_cols.items()}
)
master["combined_score"] = z_frame.mean(axis=1, skipna=True)
master.sort_values("combined_score", ascending=False)

# %% [markdown]
# ## Pick candidates
#
# - Best individual model (highest combined score, excluding consensus row).
# - Best consensus model (the row from `08_consensus_ensemble`, if present).
# - Tie-break by lower `stab_std`.

# %%
ind_rows = master[~master["notebook"].str.startswith("08_")].copy()
ind_rows = ind_rows.sort_values(["combined_score", "stab_std"], ascending=[False, True]).reset_index(
    drop=True
)
best_individual = ind_rows.iloc[0]
print("Best individual:", best_individual["notebook"], "-", best_individual["algorithm"])

cons_rows = master[master["notebook"].str.startswith("08_")]
best_consensus = cons_rows.iloc[0] if not cons_rows.empty else None
if best_consensus is not None:
    print("Best consensus:", best_consensus["algorithm"])

# %% [markdown]
# ## Build submissions

# %%
df = load_raw()
sub_template = load_sample_submission()
raw_id_column = "Id" if "Id" in df.columns else "id"
ids = df[raw_id_column].to_numpy()
sub_template_ids = sub_template["Id"].to_numpy()
assert np.array_equal(np.sort(ids), np.sort(sub_template_ids)), (
    "Id mismatch between data.csv and sample_submission.csv"
)

labels_individual = np.load(best_individual["labels_path"])
assert labels_individual.shape[0] == len(ids)
sub1_path = Path(f"submissions/01_{best_individual['notebook'].split('_', 1)[1]}.csv")
write_submission(ids, labels_individual, sub1_path)
print("Wrote", sub1_path)

if best_consensus is not None:
    labels_consensus = np.load(best_consensus["labels_path"])
    assert labels_consensus.shape[0] == len(ids)
    sub2_path = Path("submissions/02_consensus.csv")
    write_submission(ids, labels_consensus, sub2_path)
    print("Wrote", sub2_path)

# %% [markdown]
# ## Persist a final-selection record

# %%
selection = {
    "notebook": "09_final_selection",
    "candidates": {
        "individual": best_individual.to_dict(),
    },
    "submissions": [str(sub1_path)],
}
if best_consensus is not None:
    selection["candidates"]["consensus"] = best_consensus.to_dict()
    selection["submissions"].append(str(sub2_path))

write_result_json("runs/09_final", selection)

# %% [markdown]
# ## Master comparison table -> CSV (for RESULTS.md)

# %%
Path("runs/09_final").mkdir(parents=True, exist_ok=True)
master.to_csv("runs/09_final/master_comparison.csv", index=False)
master
