from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.domain_metadata import format_target_label


def format_label(value: Any, target_name: str, predicted: bool = False) -> str:
    return format_target_label(value, target_name, predicted=predicted)


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

        if probabilities is not None and target_name in {"good_review", "HeartDisease"}:
            try:
                probability_key = (
                    "probability_good_review"
                    if target_name == "good_review"
                    else "probability_heart_disease"
                )
                item[probability_key] = round(float(probabilities[position]), 4)
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
