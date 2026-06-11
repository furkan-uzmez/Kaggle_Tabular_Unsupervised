"""Column-type inference and the `build_preprocessor` strategy factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

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


ContinuousScaler = Literal["standard", "robust", "quantile", "none"]
CategoricalEncoder = Literal["onehot", "ordinal", "target_freq"]


@dataclass(frozen=True)
class PreprocessStrategy:
    """Declarative configuration for a preprocessing pipeline."""

    continuous_scaler: ContinuousScaler = "standard"
    categorical_encoder: CategoricalEncoder = "onehot"
    power_transform: bool = False


class _FrequencyEncoder:
    """Replace each category with its training-set frequency.

    Used as a pseudo target encoder when no labels are available. Implemented
    by hand to keep sklearn API (fit / transform) without an extra dependency.
    """

    def __init__(self) -> None:
        self._freqs: dict[str, dict[object, float]] = {}
        self._feature_names: list[str] = []

    @staticmethod
    def _to_frame(X: pd.DataFrame | np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=columns)

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "_FrequencyEncoder":  # noqa: D401
        X_frame = self._to_frame(X)
        self._feature_names = [str(col) for col in X_frame.columns]
        self._freqs = {}
        for col in X_frame.columns:
            counts = X_frame[col].value_counts(dropna=False, normalize=True)
            self._freqs[str(col)] = counts.to_dict()
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_frame = self._to_frame(X, self._feature_names)
        out = np.zeros((len(X_frame), len(X_frame.columns)), dtype=float)
        for j, col in enumerate(X_frame.columns):
            mapping = self._freqs[col]
            out[:, j] = X_frame[col].map(mapping).fillna(0.0).to_numpy()
        return out

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):  # noqa: D401
        return np.asarray(self._feature_names, dtype=object)


def _make_continuous_pipeline(strategy: PreprocessStrategy) -> Pipeline:
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    scaler_name = strategy.continuous_scaler
    if scaler_name == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaler_name == "robust":
        steps.append(("scale", RobustScaler()))
    elif scaler_name == "quantile":
        steps.append(("scale", QuantileTransformer(output_distribution="normal", random_state=42)))
    elif scaler_name == "none":
        pass
    else:
        raise ValueError(f"Unknown continuous_scaler: {scaler_name!r}")

    if strategy.power_transform:
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))

    return Pipeline(steps)


def _make_categorical_pipeline(strategy: PreprocessStrategy) -> Pipeline:
    impute = SimpleImputer(strategy="constant", fill_value="missing")
    encoder_name = strategy.categorical_encoder
    if encoder_name == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoder_name == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif encoder_name == "target_freq":
        encoder = _FrequencyEncoder()
    else:
        raise ValueError(f"Unknown categorical_encoder: {encoder_name!r}")
    return Pipeline([("impute", impute), ("encode", encoder)])


def build_preprocessor(strategy: PreprocessStrategy, types: ColumnTypes) -> ColumnTransformer:
    """Build a ColumnTransformer for the given strategy and column types.

    Output is always a dense ndarray (sklearn handles sparse->dense via
    `sparse_threshold=0`).
    """
    transformers: list[tuple[str, object, list[str]]] = []
    if types.continuous:
        transformers.append(("cont", _make_continuous_pipeline(strategy), types.continuous))
    if types.categorical:
        transformers.append(("cat", _make_categorical_pipeline(strategy), types.categorical))
    if not transformers:
        raise ValueError("ColumnTypes has neither continuous nor categorical columns.")
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
