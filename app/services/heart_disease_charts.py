from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.services.domain_metadata import format_target_label


HEART_TARGET = "HeartDisease"
HEART_NUMERIC_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", HEART_TARGET]


def enhance_heart_disease_report_charts(
    report: Dict[str, Any],
    dataframe: pd.DataFrame | None,
) -> None:
    """Attach additional chart-ready Heart Disease analysis views to a report."""

    target_metadata = report.get("target_metadata", {})
    if target_metadata.get("target_column") != HEART_TARGET:
        return

    prediction_charts = report.setdefault("prediction_results", {}).setdefault("charts", [])
    audit_charts = report.setdefault("model_audit", {}).setdefault("charts", [])
    eda_charts = report.setdefault("eda", {}).setdefault("charts", [])

    _annotate_existing_heart_charts(eda_charts, prediction_charts, audit_charts)

    feature_chart = _feature_importance_ranking_chart(report)
    if feature_chart:
        _prepend_unique(prediction_charts, feature_chart)
        _remove_chart(audit_charts, "heart_feature_importance_ranking")
        _remove_chart(audit_charts, "feature_importance")

    confusion_chart = _confusion_matrix_chart(report)
    if confusion_chart:
        _remove_chart(prediction_charts, "heart_confusion_matrix")
        _prepend_unique(audit_charts, confusion_chart)

    if dataframe is not None and not dataframe.empty:
        correlation_chart = _correlation_heatmap_chart(dataframe)
        if correlation_chart:
            _prepend_unique(eda_charts, correlation_chart)

        slope_chart = _st_slope_by_outcome_chart(dataframe)
        if slope_chart:
            _insert_after(eda_charts, slope_chart, after_id="heart_correlation_heatmap")


def _feature_importance_ranking_chart(report: Dict[str, Any]) -> Dict[str, Any] | None:
    feature_data = _find_chart_data(report.get("model_audit", {}).get("charts", []), "feature_importance")
    if not feature_data:
        return None

    data = []
    for rank, item in enumerate(feature_data[:10], start=1):
        feature = item.get("feature")
        value = item.get("value", item.get("importance"))
        if feature is None or value is None:
            continue
        data.append({"feature": str(feature), "value": float(value), "rank": rank})

    if not data:
        return None

    return {
        "id": "heart_feature_importance_ranking",
        "title": "Feature Importance Ranking",
        "type": "bar",
        "kind": "feature_importance",
        "priority": 100,
        "description": "Top model features associated with the HeartDisease classification outcome.",
        "data": data,
        "x": "feature",
        "y": "value",
    }


def _confusion_matrix_chart(report: Dict[str, Any]) -> Dict[str, Any] | None:
    matrix = report.get("model_audit", {}).get("confusion_matrix", [])
    if not matrix or not isinstance(matrix, list):
        return None

    labels = [
        format_target_label(0, HEART_TARGET),
        format_target_label(1, HEART_TARGET),
    ]
    data: List[Dict[str, Any]] = []
    for actual_index, row in enumerate(matrix[:2]):
        if not isinstance(row, list):
            continue
        for predicted_index, value in enumerate(row[:2]):
            actual_label = labels[actual_index] if actual_index < len(labels) else f"Actual {actual_index}"
            predicted_label = labels[predicted_index] if predicted_index < len(labels) else f"Predicted {predicted_index}"
            data.append(
                {
                    "actual": actual_label,
                    "predicted": predicted_label,
                    "value": int(value),
                }
            )

    if not data:
        return None

    return {
        "id": "heart_confusion_matrix",
        "title": "Confusion Matrix",
        "type": "heatmap",
        "kind": "confusion_matrix",
        "priority": 90,
        "description": "Validation split confusion matrix showing predicted versus actual classes.",
        "data": data,
        "x": "predicted",
        "y": "actual",
        "color": "value",
    }


def _correlation_heatmap_chart(dataframe: pd.DataFrame) -> Dict[str, Any] | None:
    available_columns = [column for column in HEART_NUMERIC_COLUMNS if column in dataframe.columns]
    if HEART_TARGET not in available_columns or len(available_columns) < 2:
        return None

    numeric_df = dataframe[available_columns].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_df.corr(numeric_only=True).round(3)
    if correlation.empty:
        return None

    data: List[Dict[str, Any]] = []
    for left in correlation.index:
        for right in correlation.columns:
            value = correlation.loc[left, right]
            if pd.isna(value):
                continue
            data.append(
                {
                    "feature_x": str(left),
                    "feature_y": str(right),
                    "value": round(float(value), 3),
                    "absolute_value": round(abs(float(value)), 3),
                }
            )

    if not data:
        return None

    return {
        "id": "heart_correlation_heatmap",
        "title": "Feature Correlation Heatmap",
        "type": "heatmap",
        "kind": "correlation_heatmap",
        "priority": 80,
        "description": "Pairwise correlations among numeric features and the HeartDisease target.",
        "data": data,
        "x": "feature_x",
        "y": "feature_y",
        "color": "value",
    }


def _st_slope_by_outcome_chart(dataframe: pd.DataFrame) -> Dict[str, Any] | None:
    if "ST_Slope" not in dataframe.columns or HEART_TARGET not in dataframe.columns:
        return None

    subset = dataframe[["ST_Slope", HEART_TARGET]].dropna().copy()
    if subset.empty:
        return None

    grouped = (
        subset.groupby(["ST_Slope", HEART_TARGET])
        .size()
        .reset_index(name="value")
        .sort_values(["ST_Slope", HEART_TARGET])
    )
    data = []
    for _, row in grouped.iterrows():
        outcome = format_target_label(row[HEART_TARGET], HEART_TARGET)
        category = str(row["ST_Slope"])
        data.append(
            {
                "category": category,
                "outcome": outcome,
                "value": int(row["value"]),
            }
        )

    if not data:
        return None

    return {
        "id": "heart_st_slope_by_outcome",
        "title": "ST_Slope by Heart Disease Outcome",
        "type": "grouped_bar",
        "kind": "categorical_outcome_distribution",
        "priority": 70,
        "description": "Distribution of ST_Slope categories across HeartDisease classes.",
        "data": data,
        "x": "category",
        "y": "value",
        "color": "outcome",
    }


def _annotate_existing_heart_charts(
    eda_charts: List[Dict[str, Any]],
    prediction_charts: List[Dict[str, Any]],
    audit_charts: List[Dict[str, Any]],
) -> None:
    for chart in eda_charts:
        chart_id = chart.get("id")
        title = str(chart.get("title", "")).lower()
        if chart_id == "class_distribution" or "heart disease distribution" in title:
            chart.setdefault("kind", "class_distribution")
            chart.setdefault("priority", 50)
        elif chart_id == "missing_values":
            chart.setdefault("kind", "missing_values")
            chart.setdefault("priority", 10)
        elif chart_id == "numeric_columns":
            chart.setdefault("kind", "numeric_summary")
            chart.setdefault("priority", 5)

    for chart in prediction_charts:
        if chart.get("id") == "prediction_distribution":
            chart.setdefault("kind", "prediction_distribution")
            chart.setdefault("priority", 40)

    for chart in audit_charts:
        if chart.get("id") == "feature_importance":
            chart.setdefault("kind", "feature_importance")
            chart.setdefault("priority", 60)



def _find_chart_data(charts: List[Dict[str, Any]], chart_id: str) -> List[Dict[str, Any]]:
    for chart in charts:
        if chart.get("id") == chart_id:
            return list(chart.get("data") or [])
    return []


def _prepend_unique(charts: List[Dict[str, Any]], chart: Dict[str, Any]) -> None:
    charts[:] = [item for item in charts if item.get("id") != chart.get("id")]
    charts.insert(0, chart)


def _remove_chart(charts: List[Dict[str, Any]], chart_id: str) -> None:
    charts[:] = [item for item in charts if item.get("id") != chart_id]


def _insert_after(charts: List[Dict[str, Any]], chart: Dict[str, Any], after_id: str) -> None:
    charts[:] = [item for item in charts if item.get("id") != chart.get("id")]
    insert_at = 0
    for index, item in enumerate(charts):
        if item.get("id") == after_id:
            insert_at = index + 1
            break
    charts.insert(insert_at, chart)
