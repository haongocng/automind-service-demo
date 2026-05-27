from __future__ import annotations

from typing import Any, Dict, List


def make_bar_chart(
    chart_id: str,
    title: str,
    description: str,
    data: List[Dict[str, Any]],
    x: str,
    y: str = "value",
) -> Dict[str, Any]:
    """Create a frontend-friendly bar chart specification."""

    return {
        "id": chart_id,
        "title": title,
        "type": "bar",
        "description": description,
        "data": data,
        "x": x,
        "y": y,
    }


def class_distribution_chart(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    return make_bar_chart(
        chart_id="class_distribution",
        title="Good vs Bad Review Distribution",
        description="Distribution of transformed review labels.",
        data=data,
        x="label",
    )


def missing_values_chart(missing_values: Dict[str, int]) -> Dict[str, Any]:
    data = [
        {"column": column, "value": int(value)}
        for column, value in missing_values.items()
        if int(value) > 0
    ]
    return make_bar_chart(
        chart_id="missing_values",
        title="Missing Values by Column",
        description="Columns with missing values. Empty data means no missing values were detected.",
        data=data,
        x="column",
    )


def feature_importance_chart(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    return make_bar_chart(
        chart_id="feature_importance",
        title="Top Feature Importance",
        description="Most influential features used by the selected model.",
        data=data,
        x="feature",
    )


def prediction_distribution_chart(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    return make_bar_chart(
        chart_id="prediction_distribution",
        title="Prediction Distribution",
        description="Distribution of predicted labels on the validation split.",
        data=data,
        x="label",
    )


def numeric_summary_chart(dataset_summary: Dict[str, Any]) -> Dict[str, Any]:
    numeric_columns = dataset_summary.get("numeric_columns", [])
    data = [{"column": column, "value": index + 1} for index, column in enumerate(numeric_columns)]
    return make_bar_chart(
        chart_id="numeric_columns",
        title="Numeric Feature Columns",
        description="Numeric columns available to the analysis pipeline.",
        data=data,
        x="column",
    )
