from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


TEXT_COLUMN_HINTS = ("comment", "message", "description", "title", "text")
ID_COLUMN_HINTS = ("_id", "id")


def profile_dataframe(
    df: pd.DataFrame,
    target_column: str | None,
    ignore_columns: List[str] | None = None,
    high_cardinality_limit: int = 50,
) -> Tuple[Dict[str, Any], List[str]]:
    """Compute lightweight data profile and suitability warnings."""

    warnings: List[str] = []
    ignore = set(ignore_columns or [])
    rows, columns = df.shape

    if rows == 0:
        warnings.append("Input data is empty.")
    if rows < 30:
        warnings.append("Dataset has fewer than 30 rows; model metrics may be unstable.")
    if columns > 100:
        warnings.append("Dataset has more than 100 columns; consider narrowing the WrenAI query.")
    if target_column and target_column not in df.columns:
        warnings.append(f"Target column '{target_column}' is missing.")

    missing_values = df.isna().sum().astype(int).to_dict()
    for column, missing_count in missing_values.items():
        if rows and missing_count / rows > 0.5:
            warnings.append(f"Column '{column}' has more than 50% missing values.")

    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    unsupported_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
        and column not in categorical_columns
        and not pd.api.types.is_datetime64_any_dtype(df[column])
    ]

    if unsupported_columns:
        warnings.append(
            "Unsupported columns will be ignored: " + ", ".join(unsupported_columns[:10])
        )

    for column in categorical_columns:
        if column in ignore:
            continue
        try:
            unique_count = int(df[column].nunique(dropna=True))
        except TypeError:
            warnings.append(f"Column '{column}' contains nested/unhashable values and will be ignored.")
            continue
        if unique_count > high_cardinality_limit:
            warnings.append(
                f"Categorical column '{column}' has high cardinality ({unique_count}); it may be dropped."
            )

    summary = {
        "rows": int(rows),
        "columns": int(columns),
        "column_names": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": missing_values,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
    return summary, warnings


def looks_like_text_column(column: str) -> bool:
    lowered = column.lower()
    return any(hint in lowered for hint in TEXT_COLUMN_HINTS)


def looks_like_id_column(column: str) -> bool:
    lowered = column.lower()
    return lowered == "id" or lowered.endswith(ID_COLUMN_HINTS)
