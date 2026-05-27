from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.profiler import looks_like_id_column, looks_like_text_column


@dataclass
class PreprocessingResult:
    X: pd.DataFrame
    preprocessor: ColumnTransformer
    numeric_features: List[str]
    categorical_features: List[str]
    dropped_columns: List[str]
    warnings: List[str]


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    exclude_columns: List[str],
    feature_columns: Optional[List[str]] = None,
    high_cardinality_limit: int = 50,
) -> PreprocessingResult:
    """Select safe feature columns and build a sklearn preprocessor."""

    warnings: List[str] = []
    dropped_columns: List[str] = []

    if feature_columns:
        candidate_columns = [column for column in feature_columns if column in df.columns]
        missing_features = sorted(set(feature_columns) - set(candidate_columns))
        if missing_features:
            warnings.append("Requested feature columns are missing: " + ", ".join(missing_features))
    else:
        candidate_columns = list(df.columns)

    forced_drop = set(exclude_columns + [target_column])
    safe_columns: List[str] = []

    for column in candidate_columns:
        if column in forced_drop:
            dropped_columns.append(column)
            continue
        if looks_like_text_column(column):
            dropped_columns.append(column)
            warnings.append(f"Dropped raw text-like column '{column}' to avoid leakage/noise in v1.")
            continue
        if looks_like_id_column(column):
            dropped_columns.append(column)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            dropped_columns.append(column)
            warnings.append(f"Dropped datetime column '{column}' in v1.")
            continue
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            unique_count = int(df[column].nunique(dropna=True))
            if unique_count > high_cardinality_limit:
                dropped_columns.append(column)
                warnings.append(
                    f"Dropped high-cardinality categorical column '{column}' ({unique_count} unique values)."
                )
                continue
        safe_columns.append(column)

    X = df[safe_columns].copy()
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    ignored_columns = [
        column
        for column in X.columns
        if column not in numeric_features and column not in categorical_features
    ]
    if ignored_columns:
        X = X.drop(columns=ignored_columns)
        dropped_columns.extend(ignored_columns)
        warnings.append("Dropped unsupported feature columns: " + ", ".join(ignored_columns))

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        warnings.append("No usable feature columns remained after preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return PreprocessingResult(
        X=X,
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_columns=sorted(set(dropped_columns)),
        warnings=warnings,
    )


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
