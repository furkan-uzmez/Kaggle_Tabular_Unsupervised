"""Raw data IO. Nothing here transforms the data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path("data/raw")
DATA_FILE = "data.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"


def load_raw(raw_dir: Path | str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the competition `data.csv` as a DataFrame.

    Args:
        raw_dir: directory containing `data.csv`. Defaults to `data/raw`.

    Returns:
        DataFrame with all original columns (including `Id`).

    Raises:
        FileNotFoundError: if `data.csv` is missing. The message points the
            user at `scripts/download_data.sh`.
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `bash scripts/download_data.sh` first.")
    return pd.read_csv(path)


def load_sample_submission(raw_dir: Path | str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the sample submission CSV (for `Id` column reference)."""
    raw_dir = Path(raw_dir)
    path = raw_dir / SAMPLE_SUBMISSION_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `bash scripts/download_data.sh` first.")
    return pd.read_csv(path)
