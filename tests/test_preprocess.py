"""Smoke tests for preprocess module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clustering.preprocess import (
    ColumnTypes,
    PreprocessStrategy,
    build_preprocessor,
    infer_column_types,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Id": np.arange(50),
            "cont_a": rng.normal(size=50),
            "cont_b": rng.exponential(size=50),
            "cat_a": rng.integers(0, 3, size=50),  # 3 unique -> categorical
            "cat_b": rng.choice(["x", "y", "z"], size=50),  # string -> categorical
        }
    )


def test_infer_column_types_basic(toy_df: pd.DataFrame) -> None:
    types = infer_column_types(toy_df)
    assert isinstance(types, ColumnTypes)
    assert set(types.continuous) == {"cont_a", "cont_b"}
    assert set(types.categorical) == {"cat_a", "cat_b"}
    assert types.excluded == ["Id"]


def test_infer_column_types_excludes_lowercase_id() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id": np.arange(100), "a": rng.normal(size=100)})
    types = infer_column_types(df)
    assert types.excluded == ["id"]
    assert types.continuous == ["a"]


def test_infer_column_types_all_continuous() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    types = infer_column_types(df)
    assert set(types.continuous) == {"a", "b"}
    assert types.categorical == []


def test_infer_column_types_all_categorical() -> None:
    df = pd.DataFrame({"a": ["x"] * 30 + ["y"] * 20, "b": [1, 2, 3] * 16 + [1, 2]})
    types = infer_column_types(df)
    assert set(types.categorical) == {"a", "b"}
    assert types.continuous == []


STRATEGIES = [
    PreprocessStrategy(continuous_scaler="standard", categorical_encoder="onehot"),
    PreprocessStrategy(continuous_scaler="robust", categorical_encoder="ordinal"),
    PreprocessStrategy(continuous_scaler="quantile", categorical_encoder="target_freq"),
    PreprocessStrategy(
        continuous_scaler="standard",
        categorical_encoder="onehot",
        power_transform=True,
    ),
    PreprocessStrategy(continuous_scaler="none", categorical_encoder="ordinal"),
]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_build_preprocessor_runs(toy_df: pd.DataFrame, strategy: PreprocessStrategy) -> None:
    types = infer_column_types(toy_df)
    transformer = build_preprocessor(strategy, types)
    X = transformer.fit_transform(toy_df.drop(columns=types.excluded))

    # Dense ndarray expected
    if hasattr(X, "toarray"):
        X = X.toarray()
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(toy_df)
    assert not np.isnan(X).any(), "Output must not contain NaNs"
