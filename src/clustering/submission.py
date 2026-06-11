"""Submission CSV writer + experiment result JSON writer."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_submission(
    ids: np.ndarray | pd.Series,
    labels: np.ndarray,
    path: Path | str,
) -> Path:
    """Write a competition-format submission CSV.

    Format:
        Id,Predicted
        0,2
        1,1
        ...
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Id": np.asarray(ids), "Predicted": np.asarray(labels, dtype=int)})
    df.to_csv(path, index=False)
    return path


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, numpy scalars, and Paths to JSON types."""
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_result_json(
    out_dir: Path | str,
    payload: dict,
    *,
    filename: str = "result.json",
    add_timestamp: bool = True,
) -> Path:
    """Serialize an experiment payload to `<out_dir>/<filename>`.

    Adds a UTC timestamp under the `timestamp` key when `add_timestamp=True`
    (unless the key already exists).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    if add_timestamp and "timestamp" not in payload:
        payload["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = out_dir / filename
    with path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=False)
    return path
