from __future__ import annotations

from typing import Any, Dict, Optional


def get_target_metadata(
    target_column: str,
    task_label: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Return domain labels used by reports, charts, and compact LLM summaries."""

    if target_column == "good_review":
        return {
            "domain_name": "E-commerce",
            "target_column": "good_review",
            "positive_label": "Good review",
            "negative_label": "Bad review",
            "distribution_title": "Good vs Bad Review Distribution",
            "target_description": "Distribution of transformed review labels.",
            "task_type": task_type or "classification",
        }

    if target_column == "HeartDisease":
        return {
            "domain_name": "Heart Disease",
            "target_column": "HeartDisease",
            "positive_label": "Heart disease",
            "negative_label": "No heart disease",
            "distribution_title": "Heart Disease Distribution",
            "target_description": "Distribution of heart disease labels in the prepared dataset.",
            "task_type": task_type or "classification",
        }

    domain_name = task_label or target_column or "Prediction"
    return {
        "domain_name": domain_name,
        "target_column": target_column,
        "positive_label": "Class 1",
        "negative_label": "Class 0",
        "distribution_title": f"{target_column} Distribution" if target_column else "Target Distribution",
        "target_description": "Distribution of target labels.",
        "task_type": task_type or "classification",
    }


def format_target_label(
    value: Any,
    target_column: str,
    predicted: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    prefix = "Predicted " if predicted else ""
    target_metadata = metadata or get_target_metadata(target_column)

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return f"{prefix}{value}"

    if numeric_value == 1:
        return f"{prefix}{target_metadata.get('positive_label', 'Class 1')}"
    if numeric_value == 0:
        return f"{prefix}{target_metadata.get('negative_label', 'Class 0')}"
    return f"{prefix}Class {numeric_value}"
