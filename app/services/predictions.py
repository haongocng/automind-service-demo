from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def format_label(value: Any, target_name: str, predicted: bool = False) -> str:
    prefix = "Predicted " if predicted else ""
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return f"{prefix}{value}"

    if target_name == "good_review":
        return f"{prefix}{'Good review' if numeric_value == 1 else 'Bad review'}"
    return f"{prefix}Class {numeric_value}"


def build_sample_predictions(
    metadata: Optional[pd.DataFrame],
    y_true: pd.Series,
    predictions,
    probabilities,
    target_name: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Build a compact prediction sample table for the report."""

    samples: List[Dict[str, Any]] = []
    true_series = pd.Series(y_true)
    pred_series = pd.Series(predictions, index=true_series.index)

    for position, row_index in enumerate(true_series.index[:limit]):
        item: Dict[str, Any] = {
            "row_index": int(row_index) if _is_int_like(row_index) else str(row_index),
            "actual": format_label(true_series.loc[row_index], target_name),
            "predicted": format_label(pred_series.loc[row_index], target_name),
        }

        if metadata is not None and row_index in metadata.index:
            order_id = metadata.loc[row_index].get("order_id") if "order_id" in metadata.columns else None
            if pd.notna(order_id):
                item["order_id"] = str(order_id)

        if probabilities is not None and target_name == "good_review":
            try:
                item["probability_good_review"] = round(float(probabilities[position]), 4)
            except (TypeError, ValueError, IndexError):
                pass

        samples.append(item)

    return samples


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
