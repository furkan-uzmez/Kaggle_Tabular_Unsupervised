# Kaggle TPS Jul 2022 -- Unsupervised Clustering

Unsupervised clustering workflow for Kaggle's
**Tabular Playground Series - Jul 2022** simulated manufacturing-control data.
The project focuses on reproducible model comparison, internal cluster-quality
metrics, and stability checks rather than leaderboard-driven tuning.

Kaggle ARI is recorded only after final candidate selection as a post-mortem
learning signal.

## Features

- Reusable `src/clustering` package for preprocessing, clustering, evaluation,
  dimensionality reduction, visualization, and submission writing.
- Numbered Jupytext notebooks covering EDA through final candidate selection.
- Committed experiment summaries under `runs/` for reproducibility.
- Local-only raw Kaggle data and submission CSVs via `.gitignore`.

## Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Kaggle CLI (`kaggle`) installed on the host
- Kaggle API token at `~/.kaggle/kaggle.json` with file mode `600`
- Kaggle competition rules accepted in the web UI

Install optional command-line prerequisites if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
pipx install kaggle
```

## Quick Start

```bash
uv sync
uv run pytest -v
uv run python -c "import clustering; print(clustering.__version__)"
```

Expected success signal:

- dependency sync completes
- tests pass
- package import prints the project version

The first `uv sync` may take 5-10 minutes because UMAP depends on numba.

## Data Access

Download the Kaggle competition files after accepting the competition rules:

```bash
bash scripts/download_data.sh
```

Expected files:

```text
data/raw/data.csv
data/raw/sample_submission.csv
```

Raw data is downloaded into `data/`, which is gitignored. Do not commit Kaggle
raw data or local submission CSVs.

## Notebook Workflow

Notebooks are stored as Jupytext percent-format `.py` files. The `.py` files
are the source of truth; generated `.ipynb` files are gitignored.

Run one notebook:

```bash
uv run jupytext --execute notebooks/01_eda.py
```

Open interactively:

```bash
uv run jupyter lab
```

Recommended execution order:

1. `notebooks/01_eda.py`
2. `notebooks/02_preprocess_experiments.py`
3. `notebooks/03_baseline_kmeans.py`
4. `notebooks/04_gmm_bgmm.py`
5. `notebooks/05_density_based.py`
6. `notebooks/06_agglomerative_spectral.py`
7. `notebooks/07_dim_reduction_clustering.py`
8. `notebooks/08_consensus_ensemble.py`
9. `notebooks/09_final_selection.py`

Some heavier notebooks support bounded local profiles through environment
variables:

```bash
KMEANS_K_MAX=12 KMEANS_STABILITY_N_ITER=3 uv run jupytext --execute notebooks/03_baseline_kmeans.py
GMM_K_MAX=8 GMM_N_INIT=1 uv run jupytext --execute notebooks/04_gmm_bgmm.py
COASSOC_MAX_N=5000 uv run jupytext --execute notebooks/08_consensus_ensemble.py
```

## Configuration

| Setting | Required | Default | Description |
| --- | --- | --- | --- |
| `~/.kaggle/kaggle.json` | yes, for download | - | Kaggle API token; keep outside the repo and set mode `600`. |
| `KMEANS_K_MAX` | no | notebook-defined | Limits KMeans scan size for local runs. |
| `KMEANS_STABILITY_N_ITER` | no | notebook-defined | Limits KMeans bootstrap stability iterations. |
| `GMM_K_MAX` | no | notebook-defined | Limits GMM/BGMM candidate cluster counts. |
| `GMM_N_INIT` | no | notebook-defined | Limits GMM initialization count. |
| `COASSOC_MAX_N` | no | notebook-defined | Bounds consensus co-association computation size. |

## Outputs

| Path | Committed | Description |
| --- | --- | --- |
| `runs/*/result.json` | yes | Experiment summaries and selected parameters. |
| `runs/*/*.npy` | yes | Derived cluster-label artifacts for downstream consensus and final selection. |
| `runs/09_final/master_comparison.csv` | yes | Aggregated model comparison table. |
| `submissions/RESULTS.md` | yes | Final candidate notes and post-mortem results. |
| `submissions/*.csv` | no | Local Kaggle submission files. |
| `data/` | no | Downloaded raw Kaggle competition data. |

The committed `runs/*/*.npy` files do not contain raw Kaggle input data. They
are derived labels kept intentionally so downstream notebooks can be reproduced
without rerunning every expensive clustering step.

## Evaluation Doctrine

Model selection uses internal unsupervised criteria:

- Silhouette score
- Davies-Bouldin score
- Calinski-Harabasz score
- BIC for GMM/BGMM
- Bootstrap stability, measured as ARI against full-data reference labels on
  repeated subsamples

Kaggle leaderboard score is not used to select or tune models.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `kaggle` command not found | Kaggle CLI is not installed or not on `PATH`. | Run `pipx install kaggle` and reopen the shell. |
| `403` while downloading data | Competition rules are not accepted. | Accept the rules on the Kaggle competition page and rerun the script. |
| `~/.kaggle/kaggle.json not found` | Kaggle API token is missing. | Create a token from Kaggle account settings and place it at `~/.kaggle/kaggle.json`. |
| Slow first install | numba/UMAP dependencies are compiling or resolving. | Let `uv sync` finish; later syncs should be faster. |
| Notebook runtime is too high | Full scan/stability settings are expensive. | Use the bounded environment variables shown above. |

## Reproducibility

- Python version is pinned via `.python-version`.
- Dependencies are locked in `uv.lock`.
- Random seeds are set to `42`.
- Notebook outputs and parameters are recorded under `runs/`.
- Submission candidates are produced by `notebooks/09_final_selection.py`.

Result mapping:

- `notebooks/01_eda.py` -> `runs/01_eda/result.json`
- `notebooks/02_preprocess_experiments.py` -> `runs/02_preprocess/result.json`
- `notebooks/03_baseline_kmeans.py` -> `runs/03_kmeans/`
- `notebooks/04_gmm_bgmm.py` -> `runs/04_gmm/`
- `notebooks/05_density_based.py` -> `runs/05_density/`
- `notebooks/06_agglomerative_spectral.py` -> `runs/06_hier_spectral/`
- `notebooks/07_dim_reduction_clustering.py` -> `runs/07_dim_reduction/`
- `notebooks/08_consensus_ensemble.py` -> `runs/08_consensus/`
- `notebooks/09_final_selection.py` -> `runs/09_final/` and `submissions/*.csv`

## Project Structure

```text
src/clustering/          reusable package code
notebooks/               Jupytext .py notebooks, numbered 01-09
scripts/download_data.sh Kaggle competition data download
runs/                    committed experiment logs and derived labels
submissions/             local submission CSVs + committed RESULTS.md
data/                    downloaded raw data, gitignored
```

## Security And Release Notes

- Keep Kaggle credentials outside the repository.
- `data/`, generated `.ipynb` files, and `submissions/*.csv` are gitignored.
- A final `gitleaks` history scan was run before public-release preparation and
  reported no leaks.

## License

This project is released under the [MIT License](LICENSE).
