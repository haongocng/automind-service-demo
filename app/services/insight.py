from __future__ import annotations

from typing import Any, Dict, List


def generate_insight(
    task_label: str,
    metrics: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    warnings: List[str],
) -> str:
    """Generate a short rule-based analyst insight."""

    if not metrics:
        if warnings:
            return "Model metrics are unavailable for this run. Review the data quality warnings before interpreting the result."
        return "Model metrics are unavailable for this run."

    feature_names = [item["feature"] for item in top_features[:3]]
    if "f1" in metrics:
        opening = f"The model achieved an F1-score of {metrics['f1']:.2f} for {task_label}."
    elif "rmse" in metrics:
        opening = f"The model achieved an RMSE of {metrics['rmse']:.2f} for {task_label}."
    else:
        opening = f"The model completed the {task_label} task."

    if feature_names:
        feature_text = ", ".join(feature_names)
        return (
            f"{opening} The most influential signals were {feature_text}. "
            "These factors are useful starting points for interpreting model behavior."
        )

    return f"{opening} Feature importance was not available for the selected model."
