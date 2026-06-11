"""Column-type inference and the `build_preprocessor` strategy factory."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ID_COLUMN = "Id"
CATEGORICAL_NUNIQUE_THRESHOLD = 20  # columns with <= 20 unique values are categorical


@dataclass(frozen=True)
class ColumnTypes:
    """Result of `infer_column_types`."""

    continuous: list[str]
    categorical: list[str]
    excluded: list[str]  # e.g., the Id column


def infer_column_types(
    df: pd.DataFrame,
    id_columns: tuple[str, ...] = (ID_COLUMN,),
    nunique_threshold: int = CATEGORICAL_NUNIQUE_THRESHOLD,
) -> ColumnTypes:
    """Heuristically split columns into continuous, categorical, and excluded.

    Rules:
      - Anything in `id_columns` is excluded.
      - Non-numeric dtypes are categorical.
      - Numeric columns with <= `nunique_threshold` distinct values are categorical.
      - All other numeric columns are continuous.
    """
    excluded = [c for c in id_columns if c in df.columns]
    work = df.drop(columns=excluded)

    continuous: list[str] = []
    categorical: list[str] = []
    for col in work.columns:
        if not pd.api.types.is_numeric_dtype(work[col]):
            categorical.append(col)
            continue
        if work[col].nunique(dropna=True) <= nunique_threshold:
            categorical.append(col)
        else:
            continuous.append(col)

    return ColumnTypes(continuous=continuous, categorical=categorical, excluded=excluded)
