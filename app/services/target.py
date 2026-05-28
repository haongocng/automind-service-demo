from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from app.schemas import TargetTransform


def apply_target_transform(
    df: pd.DataFrame,
    target_column: str,
    transform: TargetTransform | None,
) -> Tuple[pd.Series, str, Dict[str, Any], List[str]]:
    """Return transformed target, target name, summary, and warnings."""

    warnings: List[str] = []
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    target = df[target_column]
    transformed_name = target_column

    if transform and transform.type == "binary_threshold":
        if transform.threshold is None or transform.operator is None:
            raise ValueError("binary_threshold requires 'operator' and 'threshold'.")
        transformed_name = transform.positive_name or f"{target_column}_binary"
        target = _apply_binary_threshold(target, transform.operator, transform.threshold)
    elif transform and transform.type != "none":
        raise ValueError(f"Unsupported target transform: {transform.type}")

    target = target.dropna()
    distribution = target.value_counts(dropna=False).sort_index().astype(int).to_dict()
    distribution = {str(key): int(value) for key, value in distribution.items()}

    if len(distribution) <= 1:
        warnings.append("Target has only one class/value after transformation.")

    summary = {
        "original_target_column": target_column,
        "transformed_target_column": transformed_name,
        "target_column": transformed_name,
        "class_distribution": distribution,
    }
    return target, transformed_name, summary, warnings


def _apply_binary_threshold(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if operator == ">=":
        return (numeric >= threshold).astype("Int64")
    if operator == ">":
        return (numeric > threshold).astype("Int64")
    if operator == "<=":
        return (numeric <= threshold).astype("Int64")
    if operator == "<":
        return (numeric < threshold).astype("Int64")
    if operator == "==":
        return (numeric == threshold).astype("Int64")
    raise ValueError(f"Unsupported binary threshold operator: {operator}")
