from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.services.domain_metadata import format_target_label


ECOM_TARGET = "good_review"
SOURCE_TARGET = "review_score"


def enhance_ecommerce_report_charts(
    report: Dict[str, Any],
    dataframe: pd.DataFrame | None,
) -> None:
    """Attach E-commerce-specific chart-ready analysis views to a report."""

    target_metadata = report.get("target_metadata", {})
    if target_metadata.get("target_column") != ECOM_TARGET:
        return

    prediction_charts = report.setdefault("prediction_results", {}).setdefault("charts", [])
    audit_charts = report.setdefault("model_audit", {}).setdefault("charts", [])
    eda_charts = report.setdefault("eda", {}).setdefault("charts", [])

    _annotate_existing_ecommerce_charts(eda_charts, prediction_charts, audit_charts)

    feature_chart = _feature_importance_ranking_chart(report)
    if feature_chart:
        _prepend_unique(prediction_charts, feature_chart)
        _remove_chart(audit_charts, "ecommerce_feature_importance_ranking")
        _remove_chart(audit_charts, "feature_importance")

    confusion_chart = _confusion_matrix_chart(report)
    if confusion_chart:
        _remove_chart(prediction_charts, "ecommerce_confusion_matrix")
        _prepend_unique(audit_charts, confusion_chart)

    if dataframe is not None and not dataframe.empty:
        late_delivery_chart = _late_delivery_by_outcome_chart(dataframe)
        if late_delivery_chart:
            _prepend_unique(eda_charts, late_delivery_chart)

        delivery_days_chart = _delivery_days_by_outcome_chart(dataframe)
        if delivery_days_chart:
            _insert_after(eda_charts, delivery_days_chart, after_id="ecommerce_late_delivery_by_outcome")

        payment_type_chart = _payment_type_by_outcome_chart(dataframe)
        if payment_type_chart:
            _insert_after(eda_charts, payment_type_chart, after_id="ecommerce_delivery_days_by_outcome")

        freight_chart = _freight_value_by_outcome_chart(dataframe)
        if freight_chart:
            _insert_after(eda_charts, freight_chart, after_id="ecommerce_payment_type_by_outcome")


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
        "id": "ecommerce_feature_importance_ranking",
        "title": "Feature Importance Ranking",
        "type": "bar",
        "kind": "feature_importance",
        "priority": 100,
        "description": "Top model features associated with the Good review prediction outcome.",
        "data": data,
        "x": "feature",
        "y": "value",
    }


def _confusion_matrix_chart(report: Dict[str, Any]) -> Dict[str, Any] | None:
    matrix = report.get("model_audit", {}).get("confusion_matrix", [])
    if not matrix or not isinstance(matrix, list):
        return None

    labels = [
        format_target_label(0, ECOM_TARGET),
        format_target_label(1, ECOM_TARGET),
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
        "id": "ecommerce_confusion_matrix",
        "title": "Confusion Matrix",
        "type": "heatmap",
        "kind": "confusion_matrix",
        "priority": 90,
        "description": "Validation split confusion matrix showing predicted versus actual review outcomes.",
        "data": data,
        "x": "predicted",
        "y": "actual",
        "color": "value",
    }


def _late_delivery_by_outcome_chart(dataframe: pd.DataFrame) -> Dict[str, Any] | None:
    if "late_delivery" not in dataframe.columns or SOURCE_TARGET not in dataframe.columns:
        return None

    subset = dataframe[["late_delivery", SOURCE_TARGET]].dropna().copy()
    if subset.empty:
        return None

    subset["category"] = subset["late_delivery"].apply(_late_delivery_label)
    return _grouped_outcome_chart(
        subset,
        category_column="category",
        chart_id="ecommerce_late_delivery_by_outcome",
        title="Late Delivery by Review Outcome",
        description="Distribution of late-delivery status across Good review and Bad review outcomes.",
        priority=80,
        preferred_categories=["On-time delivery", "Late delivery"],
    )


def _delivery_days_by_outcome_chart(dataframe: pd.DataFrame) -> Dict[str, Any] | None:
    if "delivery_days" not in dataframe.columns or SOURCE_TARGET not in dataframe.columns:
        return None

    subset = dataframe[["delivery_days", SOURCE_TARGET]].dropna().copy()
    if subset.empty:
        return None

    days = pd.to_numeric(subset["delivery_days"], errors="coerce")
    subset = subset.loc[days.notna()].copy()
    if subset.empty:
        return None

    subset["category"] = days.loc[subset.index].apply(_delivery_days_bucket)
    return _grouped_outcome_chart(
        subset,
        category_column="category",
        chart_id="ecommerce_delivery_days_by_outcome",
        title="Delivery Days by Review Outcome",
        description="Bucketed delivery times compared across Good review and Bad review outcomes.",
        priority=70,
        kind="bucketed_outcome_distribution",
        preferred_categories=["0-3 days", "4-7 days", "8-14 days", "15+ days"],
    )


def _payment_type_by_outcome_chart(dataframe: pd.DataFrame) -> Dict[str, Any] | None:
    if "payment_type" not in dataframe.columns or SOURCE_TARGET not in dataframe.columns:
        return None

    subset = dataframe[["payment_type", SOURCE_TARGET]].dropna().copy()
    if subset.empty:
        return None

    subset["category"] = subset["payment_type"].astype(str)
    return _grouped_outcome_chart(
        subset,
        category_column="category",
        chart_id="ecommerce_payment_type_by_outcome",
        title="Payment Type by Review Outcome",
        description="Review outcomes compared across payment types.",
        priority=60,
    )


def _freight_value_by_outcome_chart(dataframe: pd.DataFrame) -> Dict[str, Any] | None:
    if "freight_value" not in dataframe.columns or SOURCE_TARGET not in dataframe.columns:
        return None

    subset = dataframe[["freight_value", SOURCE_TARGET]].dropna().copy()
    if subset.empty:
        return None

    freight = pd.to_numeric(subset["freight_value"], errors="coerce")
    subset = subset.loc[freight.notna()].copy()
    freight = freight.loc[subset.index]
    if subset.empty or freight.nunique() < 3:
        return None

    try:
        subset["category"] = pd.qcut(
            freight,
            q=3,
            labels=["Low freight", "Medium freight", "High freight"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        return None

    return _grouped_outcome_chart(
        subset,
        category_column="category",
        chart_id="ecommerce_freight_value_by_outcome",
        title="Freight Value by Review Outcome",
        description="Freight value buckets compared across Good review and Bad review outcomes.",
        priority=50,
        kind="bucketed_outcome_distribution",
        preferred_categories=["Low freight", "Medium freight", "High freight"],
    )


def _grouped_outcome_chart(
    dataframe: pd.DataFrame,
    category_column: str,
    chart_id: str,
    title: str,
    description: str,
    priority: int,
    kind: str = "categorical_outcome_distribution",
    preferred_categories: List[str] | None = None,
) -> Dict[str, Any] | None:
    subset = dataframe[[category_column, SOURCE_TARGET]].dropna().copy()
    if subset.empty:
        return None

    subset["outcome_value"] = (pd.to_numeric(subset[SOURCE_TARGET], errors="coerce") >= 4).astype("Int64")
    subset = subset.dropna(subset=["outcome_value"])
    if subset.empty:
        return None

    grouped = (
        subset.groupby([category_column, "outcome_value"])
        .size()
        .reset_index(name="value")
        .sort_values([category_column, "outcome_value"])
    )
    data = []
    for _, row in grouped.iterrows():
        data.append(
            {
                "category": str(row[category_column]),
                "outcome": format_target_label(row["outcome_value"], ECOM_TARGET),
                "value": int(row["value"]),
            }
        )

    if not data:
        return None

    if preferred_categories:
        order = {category: index for index, category in enumerate(preferred_categories)}
        data.sort(key=lambda row: (order.get(row["category"], len(order)), row["outcome"]))

    return {
        "id": chart_id,
        "title": title,
        "type": "grouped_bar",
        "kind": kind,
        "priority": priority,
        "description": description,
        "data": data,
        "x": "category",
        "y": "value",
        "color": "outcome",
    }


def _late_delivery_label(value: Any) -> str:
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return str(value)
    return "Late delivery" if numeric_value == 1 else "On-time delivery"


def _delivery_days_bucket(value: Any) -> str:
    numeric_value = float(value)
    if numeric_value <= 3:
        return "0-3 days"
    if numeric_value <= 7:
        return "4-7 days"
    if numeric_value <= 14:
        return "8-14 days"
    return "15+ days"


def _annotate_existing_ecommerce_charts(
    eda_charts: List[Dict[str, Any]],
    prediction_charts: List[Dict[str, Any]],
    audit_charts: List[Dict[str, Any]],
) -> None:
    for chart in eda_charts:
        chart_id = chart.get("id")
        if chart_id == "class_distribution":
            chart.setdefault("kind", "class_distribution")
            chart.setdefault("priority", 40)
        elif chart_id == "missing_values":
            chart.setdefault("kind", "missing_values")
            chart.setdefault("priority", 10)
        elif chart_id == "numeric_columns":
            chart.setdefault("kind", "numeric_summary")
            chart.setdefault("priority", 5)

    for chart in prediction_charts:
        if chart.get("id") == "prediction_distribution":
            chart.setdefault("kind", "prediction_distribution")
            chart.setdefault("priority", 45)

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
    insert_at = len(charts)
    for index, item in enumerate(charts):
        if item.get("id") == after_id:
            insert_at = index + 1
            break
    charts.insert(insert_at, chart)
