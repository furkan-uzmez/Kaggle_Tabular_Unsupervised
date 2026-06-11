# Kaggle TPS Jul 2022 -- Unsupervised Clustering

This project explores Kaggle's **Tabular Playground Series - Jul 2022**, a fully unsupervised clustering challenge on simulated manufacturing control data.

The project is intentionally **not leaderboard-driven**. Models are selected using internal clustering metrics and bootstrap stability. Kaggle ARI is recorded only after final candidate selection as a post-mortem learning signal.

## Project Structure

```text
src/clustering/          reusable package code
notebooks/               jupytext .py notebooks, numbered 01-09
scripts/download_data.sh Kaggle competition data download
runs/                    committed experiment logs (JSON/CSV/NPY)
submissions/             local submission CSVs + committed RESULTS.md
data/                    downloaded raw data (gitignored)
```

## Requirements

- Python 3.11
- uv
- Kaggle CLI (`kaggle`) installed on the host
- Kaggle API token at `~/.kaggle/kaggle.json` with mode `600`
- Competition rules accepted in the Kaggle web UI

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Kaggle CLI if needed:

```bash
pipx install kaggle
```

## Setup

```bash
uv sync
```

The first sync may take 5-10 minutes because UMAP depends on numba.

Verify package import:

```bash
uv run python -c "import clustering; print(clustering.__version__)"
```

## Download Data

```bash
bash scripts/download_data.sh
```

Expected files:

```text
data/raw/data.csv
data/raw/sample_submission.csv
```

If you see a 403 from Kaggle, accept the competition rules in the web UI first.

## Run Tests

```bash
uv run pytest -v
```

Tests cover preprocessing and evaluation sanity, not model quality.

## Notebook Workflow

Notebooks are stored as jupytext percent-format `.py` files. The `.py` files are the source of truth; `.ipynb` files are gitignored.

Run a notebook:

```bash
uv run jupytext --execute notebooks/01_eda.py
```

Open interactively:

```bash
uv run jupyter lab
```

Recommended order:

1. `notebooks/01_eda.py`
2. `notebooks/02_preprocess_experiments.py`
3. `notebooks/03_baseline_kmeans.py`
4. `notebooks/04_gmm_bgmm.py`
5. `notebooks/05_density_based.py`
6. `notebooks/06_agglomerative_spectral.py`
7. `notebooks/07_dim_reduction_clustering.py`
8. `notebooks/08_consensus_ensemble.py`
9. `notebooks/09_final_selection.py`

Some heavy notebooks support environment variables to run bounded local profiles. The committed `runs/*/result.json` files record actual ranges, subsample sizes, and stability iterations used.

Examples:

```bash
KMEANS_K_MAX=12 KMEANS_STABILITY_N_ITER=3 uv run jupytext --execute notebooks/03_baseline_kmeans.py
GMM_K_MAX=8 GMM_N_INIT=1 uv run jupytext --execute notebooks/04_gmm_bgmm.py
COASSOC_MAX_N=5000 uv run jupytext --execute notebooks/08_consensus_ensemble.py
```

## Outputs

- `runs/*/result.json`: experiment summaries, committed.
- `runs/*/*.npy`: saved labels for downstream consensus/final selection, committed.
- `runs/09_final/master_comparison.csv`: aggregated model comparison table.
- `submissions/*.csv`: local Kaggle submissions, gitignored.
- `submissions/RESULTS.md`: final candidate notes, committed.

## Evaluation Doctrine

Model selection uses:

- Silhouette score
- Davies-Bouldin score
- Calinski-Harabasz score
- BIC for GMM/BGMM
- Bootstrap stability (ARI against the full-data reference labels on repeated subsamples)

Kaggle leaderboard score is not used to select or tune models.

## Reproducibility

- Python pinned via `.python-version`
- Dependencies locked in `uv.lock`
- Random seeds set to `42`
- Notebook output logs stored under `runs/`
- Submission candidates produced by `09_final_selection.py`
