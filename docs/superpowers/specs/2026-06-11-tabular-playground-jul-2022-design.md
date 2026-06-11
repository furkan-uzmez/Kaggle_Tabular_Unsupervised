# Tabular Playground Series — Jul 2022: Design Spec

- **Date:** 2026-06-11
- **Competition:** [Tabular Playground Series - Jul 2022](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022) (Late Submission)
- **Status:** Approved design, ready for implementation planning

## 1. Goals & Non-Goals

### 1.1 Goals

1. **Learning-focused**: Systematically compare classical unsupervised clustering methods (KMeans family, GMM/BGMM, density-based, hierarchical, spectral) on a real tabular dataset.
2. **Reusable infrastructure**: Build a modular, `uv`-managed Python package (`src/clustering/`) that can be reused for future tabular clustering problems.
3. **Methodologically honest unsupervised evaluation**: Select models without using the leaderboard. Internal metrics + stability are the only decision signals.
4. **Reproducible**: Anyone can clone the repo, run `uv sync`, download the data, execute all notebooks end-to-end, and reproduce the same submissions.

### 1.2 Non-Goals (explicit YAGNI)

- ❌ Deep learning / autoencoders / Deep Embedded Clustering
- ❌ Hydra config system
- ❌ MLflow / W&B experiment tracking
- ❌ CI/CD pipeline
- ❌ Docker container
- ❌ Leaderboard-driven model selection
- ❌ Copying public-notebook techniques (we learn from our own experiments)
- ❌ Hyperparameter auto-tuning (Optuna, etc.) — systematic grid sweeps are sufficient

## 2. Problem Summary

- **Task:** Cluster simulated manufacturing control data into unknown number of control states.
- **Data:** `data.csv` (43.69 MB, 32 columns: continuous + categorical mixed; one column is `Id`). Row count is not specified by Kaggle — confirmed in `01_eda.py` after download. No labels. No train/test split — all rows are clustered.
- **Metric:** Adjusted Rand Index (ARI) against hidden ground-truth labels — only visible after submission.
- **Submission format:** `Id,Predicted` CSV, one row per input row.

## 3. Architecture

### 3.1 Repository Layout

```
Kaggle_Tabular_Unsupervised/
├── pyproject.toml              # uv-managed, PEP 621 + [dependency-groups]
├── uv.lock                     # committed
├── .python-version             # 3.11
├── .gitignore                  # data/, *.ipynb, .venv/, submissions/*.csv (RESULTS.md committed); runs/ is committed
├── README.md                   # setup + repro steps
├── src/
│   └── clustering/
│       ├── __init__.py
│       ├── data.py             # load_raw(), load_sample_submission()
│       ├── preprocess.py       # column typing, build_preprocessor(strategy)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── kmeans.py       # KMeans + MiniBatchKMeans wrappers
│       │   ├── gmm.py          # GaussianMixture + BayesianGaussianMixture
│       │   ├── density.py      # DBSCAN, HDBSCAN
│       │   ├── hierarchical.py # Agglomerative, Spectral
│       │   └── consensus.py    # co-association matrix + ensemble
│       ├── dim_reduction.py    # PCA, UMAP
│       ├── evaluate.py         # silhouette, DB, CH, BIC, bootstrap stability
│       ├── submission.py       # write_submission(labels, path)
│       └── viz.py              # shared plots (cluster scatter, elbow curves)
├── notebooks/                  # jupytext .py percent format
│   ├── 01_eda.py
│   ├── 02_preprocess_experiments.py
│   ├── 03_baseline_kmeans.py
│   ├── 04_gmm_bgmm.py
│   ├── 05_density_based.py
│   ├── 06_agglomerative_spectral.py
│   ├── 07_dim_reduction_clustering.py
│   ├── 08_consensus_ensemble.py
│   └── 09_final_selection.py
├── scripts/
│   └── download_data.sh        # kaggle competitions download
├── tests/
│   ├── test_preprocess.py      # smoke: pipeline shape/type checks
│   └── test_evaluate.py        # ARI/silhouette sanity on synthetic data
├── data/                       # .gitignore'd
│   ├── raw/                    # data.csv, sample_submission.csv
│   └── processed/              # optional intermediate outputs
├── submissions/                # CSVs gitignore'd; RESULTS.md committed
└── kaggle_pages/               # existing — competition page captures
```

### 3.2 Design Principles

- **`src/clustering/` package = single source of truth for logic.** Notebooks call into it; notebooks do not embed reusable logic. DRY + testable.
- **`models/` subpackage**: each algorithm family in its own file — single-file bloat avoided, easy to extend.
- **`dim_reduction.py` separate**: used by both 07 and 08, kept shared.
- **`evaluate.py` centralizes the evaluation doctrine**: internal metrics + bootstrap stability in one module.
- **`viz.py`**: repeated plotting code (silhouette plots, 2D cluster scatter) shared.
- **`scripts/` for shell-level ops**: keep Python and shell separated.
- **Minimal tests**: only preprocess pipeline and evaluate sanity (synthetic data). Safety net for reproducibility, not coverage chasing.

## 4. Environment & Dependencies

### 4.1 Python & uv Topology

- **Python:** 3.11 (`.python-version` pinned)
- **uv topology:** `uv init --package` — `src/` layout so notebooks can `from clustering.data import load_raw`.
- **`uv.lock` committed.** CI/other machines use `uv sync --locked`.

### 4.2 `pyproject.toml` Skeleton

```toml
[project]
name = "clustering"
version = "0.1.0"
description = "Unsupervised clustering — Kaggle TPS Jul 2022"
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "plotly>=5.18",
    "hdbscan>=0.8.33",
    "umap-learn>=0.5.5",
    "scikit-learn-extra>=0.3.0",
    "jupytext>=1.16",
]

[dependency-groups]
dev = [
    "jupyter>=1.0",
    "jupyterlab>=4.0",
    "ipykernel>=6.29",
    "pytest>=8.0",
    "ruff>=0.4",
]

[tool.jupytext]
formats = "py:percent"

[tool.ruff]
line-length = 100
target-version = "py311"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 4.3 Decisions

- **No `kagglehub` in project deps.** Data is downloaded with host-level legacy `kaggle` CLI via `scripts/download_data.sh`. Aligns with `kaggle-asset-integration` skill: competition lifecycle → legacy `kaggle`.
- **Test & lint in `dev` group.** Production install (`uv sync --no-dev`) excludes them.
- **`scikit-learn-extra` included for KMedoids.** If unused after experiments, removed in a cleanup commit.
- **Initial `uv sync` may take 5–10 min** due to numba (UMAP dep) compilation. Documented in README.

## 5. Data Pipeline

### 5.1 Acquisition

`scripts/download_data.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle competitions download -c tabular-playground-series-jul-2022 -p data/raw/
cd data/raw && unzip -o tabular-playground-series-jul-2022.zip && rm tabular-playground-series-jul-2022.zip
```

Preconditions: `kaggle` CLI installed on host, token at `~/.kaggle/kaggle.json`, competition rules accepted. Script fails loudly if any are missing.

### 5.2 Preprocessing (`preprocess.py`)

- **`infer_column_types(df)`**: excludes `Id`. Splits remaining 31 columns into `continuous` / `categorical` via dtype + `nunique() < threshold` heuristic. EDA confirms manually; override mechanism available.
- **`build_preprocessor(strategy)`**: strategy-selector factory, not a single hard-coded pipeline. Parameters:
  - `continuous_scaler`: `"standard" | "robust" | "quantile" | "none"`
  - `categorical_encoder`: `"onehot" | "ordinal" | "target_freq"` (frequency encoding since no labels exist)
  - `power_transform`: `bool` (Yeo-Johnson — important for GMM Gaussian assumption)
- **`fit_transform` on full data.** No train/test split — fully unsupervised.
- **Missing values**: continuous → median impute; categorical → "missing" sentinel. EDA decides whether any column has them.

## 6. Evaluation Doctrine

**Core rule: model selection uses internal metrics + stability only. Leaderboard score is NOT a decision input.**

### 6.1 Four Signals

| Signal | What it measures | Applicable to |
|---|---|---|
| **Silhouette** | Intra-cluster tightness vs. inter-cluster separation | All (noise points excluded for HDBSCAN) |
| **Davies-Bouldin** | Cluster separability (lower is better) | All |
| **Calinski-Harabasz** | Variance ratio (higher is better) | All |
| **BIC** | Likelihood + complexity penalty | GMM / BGMM only |
| **Stability (bootstrap ARI)** | Self-consistency across reseeds/subsamples | All (with noise-alignment for DBSCAN/HDBSCAN) |

### 6.2 Stability Protocol

`bootstrap_stability(model, X, n_iter=20, sample_frac=0.8)`:

1. Fit on full data → `labels_full`.
2. For `n_iter` iterations: subsample 80% → fit → compute ARI vs. `labels_full` on the intersecting indices.
3. Report mean ± std. High mean + low std = stable model.

Heavy models (Spectral, HDBSCAN with large data): `n_iter=5` with a note explaining the reduction.

### 6.3 n_clusters Scan Protocol

`scan_n_clusters(model_class, X, k_range)`:

- For each k: compute silhouette, DB, CH, (BIC if applicable), stability mean.
- Standardize signals → compute simple averaged "consensus score". Conflicts are reported, not hidden.
- Output: 5-curve panel plot + numeric table.

### 6.4 Final Selection Rule (notebook 09)

1. Per algorithm: pick "best k" → record all 4 signals + stability.
2. Rank candidates by **combined signal**, not single metric (silhouette alone correlates weakly with ARI).
3. Consensus clustering result evaluated as a separate candidate.
4. Tie-breaker: stability (lower variance preferred).

### 6.5 Submission Policy

- Local selection: completely ARI-free, per §6.4.
- 1–2 final candidates submitted: `submissions/01_<model>.csv`, `submissions/02_consensus.csv`.
- LB score recorded in `submissions/RESULTS.md` as a **post-mortem learning note**. It does not retroactively change model choice.

## 7. Notebook & Experiment Plan

Naming: `NN_<topic>.py` (jupytext percent format). Each notebook is self-contained but reads/writes a small JSON log under `runs/`.

### 7.1 Notebook Contents

**`01_eda.py` — Exploration**
- Load, shape, dtype, missing-value matrix.
- Continuous: histograms, KDE, boxplots, skewness, correlation heatmap.
- Categorical: value counts, cardinality, distributions.
- Confirm `infer_column_types` heuristic; note overrides.
- 2D snapshots: PCA(2) and UMAP(2) uncolored scatter — does the raw data already show structure?
- Output: top-of-notebook summary cell `eda_findings`.

**`02_preprocess_experiments.py` — Choose preprocessing strategies**
- Run 4–6 `build_preprocessor` combos.
- For each: quick `KMeans(k=10)` silhouette + PCA(2) plot.
- Identify the "most Gaussian-friendly" strategy for GMM (used in 04).
- Output: 2–3 winning strategies passed downstream.

**`03_baseline_kmeans.py` — KMeans / MiniBatchKMeans**
- `scan_n_clusters(KMeans, X, range(2, 31))` → 4-signal curves + consensus score.
- Stability for best k.
- MiniBatchKMeans comparison (speed vs. quality).
- Output: `runs/03_kmeans/result.json` + scan table CSV.

**`04_gmm_bgmm.py` — GaussianMixture + BayesianGaussianMixture**
- GMM grid: `covariance_type ∈ {full, tied, diag, spherical}` × n_components sweep.
- BIC primary signal; silhouette/DB/CH/stability confirmatory.
- BayesianGaussianMixture: `n_components=30` upper bound, Dirichlet prior → automatic k selection (count of "dead" components).
- Output: `runs/04_gmm/result.json`.

**`05_density_based.py` — DBSCAN + HDBSCAN**
- DBSCAN: k-distance graph → eps selection methodology.
- HDBSCAN: `min_cluster_size`, `min_samples` sweep; cluster persistence + condensed-tree plot.
- Noise-point ratio report.
- **Noise-point treatment decision (made here, based on EDA findings)**: either (a) assign noise to nearest cluster centroid via KNN, or (b) keep noise as its own pseudo-cluster id. Decision and rationale recorded in the notebook.
- Output: `runs/05_density/result.json`.

**`06_agglomerative_spectral.py` — Hierarchical + Spectral**
- Agglomerative: `linkage ∈ {ward, complete, average}` × n_clusters sweep.
- Dendrogram (ward).
- Spectral clustering: `affinity ∈ {nearest_neighbors, rbf}`. RBF is attempted only if the dataset has fewer than 20 000 rows after preprocessing; otherwise, the notebook fits Spectral on a 10 000-row subsample then propagates labels to the rest via 5-NN majority vote. Decision (full / subsample) recorded in the notebook.
- Output: `runs/06_hier_spectral/result.json`.

**`07_dim_reduction_clustering.py` — PCA / UMAP + clustering**
- PCA: cumulative explained variance → component count.
- UMAP: `n_components ∈ {2, 5, 10}` × `n_neighbors ∈ {15, 30, 50}`; topology-preservation visuals.
- Re-run the 2 strongest algorithms from 03–06 on each embedding.
- "Did embedding help?" comparison table (4 signals).
- Output: `runs/07_dim_reduction/result.json`.

**`08_consensus_ensemble.py` — Consensus clustering**
- Collect **best label set per notebook from 03–07** (5 label sets total: one from each of 03, 04, 05, 06, 07). If a notebook produced multiple strong candidates, the one with the highest combined signal score (§6.4) is taken.
- Build co-association matrix: `M[i,j] = (count where i,j share a cluster) / 5`.
- Apply Agglomerative (average linkage) on `1 - M` distance → consensus labels. Spectral is an optional alternative if Agglomerative is unstable.
- Score consensus with the same 4 signals + stability.
- Compare against individual models.
- Output: `runs/08_consensus/result.json`.

**`09_final_selection.py` — Final selection + submissions**
- Aggregate all `runs/*/result.json` into a master comparison table.
- Apply §6.4 rule → pick 1 individual + 1 consensus candidate.
- Write `submissions/01_<model_name>.csv`, `submissions/02_consensus.csv`.
- Write `submissions/RESULTS.md`: chosen candidates, rationale, internal metrics. LB-score column initially empty; user fills it after submission as a post-mortem note.

### 7.2 Experiment Log Format

`runs/<notebook>/result.json`:

```json
{
  "notebook": "03_baseline_kmeans",
  "timestamp": "2026-06-11T14:32:00",
  "preprocess_strategy": "standard+freq+yeo-johnson",
  "best_model": {
    "algorithm": "KMeans",
    "params": {"n_clusters": 12, "n_init": 20, "random_state": 42},
    "internal_metrics": {
      "silhouette": 0.142,
      "davies_bouldin": 1.87,
      "calinski_harabasz": 4521.3,
      "bic": null
    },
    "stability": {"mean_ari": 0.78, "std_ari": 0.04, "n_iter": 20}
  },
  "scan_summary_path": "runs/03_kmeans/scan_table.csv"
}
```

`runs/` directory is **committed** (small JSON + CSV files = experiment history is part of the deliverable).

### 7.3 Reproducibility

- All models use `random_state=42`.
- Each notebook starts with `np.random.seed(42)` + `random.seed(42)`.
- Each `result.json` records full preprocessing strategy + hyperparameters.

## 8. Testing Strategy

Minimal — smoke / sanity tests, not clustering-quality tests.

**`test_preprocess.py`** (synthetic 50×5 DataFrame):
- `infer_column_types` separates continuous/categorical correctly.
- `build_preprocessor`: every strategy combo `fit_transform`s; output is `np.ndarray`, no NaNs, row count preserved.
- Edge cases: single-column, all-categorical, all-continuous.

**`test_evaluate.py`** (`sklearn.datasets.make_blobs`):
- `bootstrap_stability`: same data + same seed → ARI ≈ 1.0.
- `scan_n_clusters`: on 3-blob synthetic data, at least one signal recovers k=3.
- All internal-metric calls execute (silhouette, DB, CH).

**Out of scope (intentionally untested):**
- Individual clustering algorithms (sklearn covers them).
- Notebook execution as a test.
- Submission CSV format (manual visual check is enough for a small file).

Run with `uv run pytest -v`. No CI, but expected before commits that touch `src/clustering/`.

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `umap-learn` numba compile slowness | High | Low (one-time) | README note: "first `uv sync` may take 5–10 min" |
| HDBSCAN slow on full 43MB data | Medium | Medium | Prototype on 10% subsample first; use `core_dist_n_jobs=-1` |
| Spectral clustering memory blowup (n²) | High | High | 06 pre-emptively subsamples + KNN-propagates; no full-data attempt |
| `scikit-learn-extra` is unmaintained | Medium | Low | Drop if KMedoids unused; fallback `pyclustering` |
| Categorical column mis-typing | Medium | High (preprocess wrong) | EDA manual confirmation + override mechanism |
| GMM convergence warnings | High | Low | Bump `reg_covar`, `n_init ≥ 5`, log warnings, don't halt |
| Stability analysis slowness (heavy models × 20 iter) | Medium | Medium | Default `n_iter=20`; heavy models `n_iter=5`, note rationale |
| Notebook `.py` ↔ `.ipynb` sync conflicts | Low | Medium | `.ipynb` in `.gitignore`; `.py` is the only source of truth; `jupytext --set-formats py:percent` in setup |
| Kaggle CLI / token missing | Medium | High (data can't download) | `scripts/download_data.sh` checks for token & gives clear error |
| `runs/` JSON schema drift over time | Medium | Low | 09 reads tolerantly, missing fields handled gracefully |
| n_clusters signals contradicting | High | Medium | Expected; record rationale in the notebook; accept "no single right answer" |
| Consensus worse than best single model | Medium | Low (still has learning value) | 09 submits both candidates; RESULTS.md records the comparison |

## 10. Definition of Done

- [ ] `uv sync` works from clean clone.
- [ ] `scripts/download_data.sh` succeeds and `data/raw/data.csv` exists.
- [ ] `uv run pytest` is green.
- [ ] All 9 notebooks execute end-to-end without error.
- [ ] Every notebook produces `runs/<notebook>/result.json`, committed.
- [ ] `09_final_selection.py` produces the master comparison table.
- [ ] `submissions/01_*.csv` and `submissions/02_consensus.csv` produced locally (CSVs themselves are gitignored).
- [ ] `submissions/RESULTS.md` is committed; records chosen candidates + rationale (LB score optional, post-mortem).
- [ ] `README.md` documents: setup, data download, notebook execution, expected outputs.
- [ ] `.gitignore` correct: `data/`, `submissions/*.csv` (RESULTS.md exempted via `!submissions/RESULTS.md`), `.ipynb`, `.venv/`, `__pycache__/`.

## 11. References

- Competition page captures: `kaggle_pages/tabular-playground-series-jul-2022/`
- Local skills: `.agent/skills/kaggle-asset-integration/`, `.agent/skills/uv-python-workflows/`
