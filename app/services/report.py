from __future__ import annotations

from typing import Any, Dict, List

from app.services.charting import (
    class_distribution_chart,
    feature_importance_chart,
    missing_values_chart,
    numeric_summary_chart,
    prediction_distribution_chart,
)


def build_prediction_report(
    title: str,
    task_label: str,
    target_definition: str,
    dataset_summary: Dict[str, Any],
    target_summary: Dict[str, Any],
    model: Dict[str, Any],
    metrics: Dict[str, Any],
    charts: Dict[str, List[Dict[str, Any]]],
    top_features: List[Dict[str, Any]],
    sample_predictions: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    """Build an AutoMind-style structured report for frontend rendering."""

    class_distribution = _named_class_distribution(target_summary.get("class_distribution", {}))
    missing_values = dataset_summary.get("missing_values", {})
    eda_charts = [
        class_distribution_chart(class_distribution),
        missing_values_chart(missing_values),
        numeric_summary_chart(dataset_summary),
    ]
    prediction_charts = [
        prediction_distribution_chart(charts.get("prediction_distribution", [])),
    ]
    audit_charts = [
        feature_importance_chart(charts.get("feature_importance", [])),
    ]

    limitations = _limitations()
    executive_summary = _executive_summary(task_label, dataset_summary, metrics, top_features, warnings)
    eda_summary = _eda_summary(dataset_summary, class_distribution, missing_values)
    key_insights = _key_insights(metrics, top_features, class_distribution, warnings)
    recommendations = _recommendations(top_features, warnings)

    report = {
        "title": title,
        "executive_summary": executive_summary,
        "dataset_overview": {
            "rows": dataset_summary.get("rows", 0),
            "columns": dataset_summary.get("columns", 0),
            "target": target_summary.get("transformed_target_column"),
            "numeric_columns": dataset_summary.get("numeric_columns", []),
            "categorical_columns": dataset_summary.get("categorical_columns", []),
            "missing_values": missing_values,
            "class_distribution": {item["label"]: item["value"] for item in class_distribution},
        },
        "eda": {
            "summary": eda_summary,
            "charts": eda_charts,
        },
        "key_insights": key_insights,
        "prediction_task": {
            "task_name": task_label,
            "target_definition": target_definition,
            "features_used": model.get("numeric_features", []) + model.get("categorical_features", []),
            "excluded_columns": model.get("dropped_columns", []),
            "selected_model": model.get("selected_model"),
            "candidate_models": model.get("candidate_models", []),
        },
        "prediction_results": {
            "sample_predictions": sample_predictions[:20],
            "charts": prediction_charts,
        },
        "model_audit": {
            "note": "Metrics are computed on a held-out validation split and are used only to audit the demo model.",
            "metrics": {key: value for key, value in metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": metrics.get("confusion_matrix", []),
            "charts": audit_charts,
        },
        "recommendations": recommendations,
        "warnings": warnings,
        "limitations": limitations,
    }
    report["report_markdown"] = render_report_markdown(report)
    return report


def render_report_markdown(report: Dict[str, Any]) -> str:
    """Render a compact markdown report from the structured object."""

    lines: List[str] = [f"## {report['title']}", ""]
    lines.append("### Executive Summary")
    lines.extend([f"- {item}" for item in report.get("executive_summary", [])])
    lines.append("")

    overview = report.get("dataset_overview", {})
    lines.append("### Dataset Overview")
    lines.append(f"- Rows: {overview.get('rows')}")
    lines.append(f"- Columns: {overview.get('columns')}")
    lines.append(f"- Target: {overview.get('target')}")
    lines.append(f"- Numeric columns: {len(overview.get('numeric_columns', []))}")
    lines.append(f"- Categorical columns: {len(overview.get('categorical_columns', []))}")
    lines.append("")

    lines.append("### EDA Summary")
    lines.extend([f"- {item}" for item in report.get("eda", {}).get("summary", [])])
    lines.append("")

    task = report.get("prediction_task", {})
    lines.append("### Prediction Task")
    lines.append(f"- Task: {task.get('task_name')}")
    lines.append(f"- Target definition: {task.get('target_definition')}")
    lines.append(f"- Selected model: {task.get('selected_model')}")
    lines.append(f"- Features used: {len(task.get('features_used', []))}")
    lines.append("")

    audit = report.get("model_audit", {})
    lines.append("### Model Audit")
    lines.append(f"- Note: {audit.get('note')}")
    for metric, value in audit.get("metrics", {}).items():
        lines.append(f"- {metric}: {value}")
    lines.append("")

    lines.append("### Key Insights")
    lines.extend([f"- {item}" for item in report.get("key_insights", [])])
    lines.append("")

    agent_insights = report.get("agent_insights")
    if agent_insights:
        lines.append("### Agent Insights")
        summary = agent_insights.get("summary")
        if summary:
            lines.append(f"- Summary: {summary}")
        for item in agent_insights.get("business_insights", []):
            lines.append(f"- Business insight: {item}")
        for item in agent_insights.get("recommendations", []):
            lines.append(f"- Recommendation: {item}")
        for item in agent_insights.get("risk_notes", []):
            lines.append(f"- Risk note: {item}")
        lines.append("")

    lines.append("### Recommendations")
    lines.extend([f"- {item}" for item in report.get("recommendations", [])])
    lines.append("")

    lines.append("### Warnings and Limitations")
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend([f"- Warning: {item}" for item in warnings])
    lines.extend([f"- Limitation: {item}" for item in report.get("limitations", [])])
    return "\n".join(lines)


def _named_class_distribution(distribution: Dict[str, int]) -> List[Dict[str, Any]]:
    if "0" in distribution or "1" in distribution:
        return [
            {"label": "Bad review", "value": int(distribution.get("0", 0))},
            {"label": "Good review", "value": int(distribution.get("1", 0))},
        ]
    return [{"label": str(label), "value": int(value)} for label, value in distribution.items()]


def _executive_summary(
    task_label: str,
    dataset_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    warnings: List[str],
) -> List[str]:
    rows = dataset_summary.get("rows", 0)
    selected_metric = f"F1-score {metrics['f1']:.2f}" if "f1" in metrics else "model metrics unavailable"
    feature_text = _feature_text(top_features)
    summary = [
        f"AutoMind analyzed {rows} row-level records for the {task_label} task.",
        f"The validation audit reports {selected_metric}; these are demo validation metrics, not production guarantees.",
    ]
    if feature_text:
        summary.append(f"The strongest model signals are {feature_text}.")
    if warnings:
        summary.append("Warnings were detected and should be reviewed before using the result operationally.")
    return summary


def _eda_summary(
    dataset_summary: Dict[str, Any],
    class_distribution: List[Dict[str, Any]],
    missing_values: Dict[str, int],
) -> List[str]:
    missing_total = sum(int(value) for value in missing_values.values())
    class_text = ", ".join(f"{item['label']}: {item['value']}" for item in class_distribution)
    return [
        f"The dataset contains {dataset_summary.get('rows', 0)} rows and {dataset_summary.get('columns', 0)} columns.",
        f"Detected {len(dataset_summary.get('numeric_columns', []))} numeric columns and {len(dataset_summary.get('categorical_columns', []))} categorical columns.",
        f"Total missing values across all columns: {missing_total}.",
        f"Target distribution: {class_text}.",
    ]


def _key_insights(
    metrics: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    class_distribution: List[Dict[str, Any]],
    warnings: List[str],
) -> List[str]:
    insights: List[str] = []
    if "f1" in metrics:
        insights.append(f"The selected model achieved an F1-score of {metrics['f1']:.2f} on the validation split.")
    if top_features:
        insights.append(f"Top predictive factors include {_feature_text(top_features)}.")
    if class_distribution:
        total = sum(item["value"] for item in class_distribution)
        if total:
            majority = max(class_distribution, key=lambda item: item["value"])
            share = majority["value"] / total
            insights.append(f"The majority class is {majority['label']} ({share:.0%} of labeled rows).")
    if warnings:
        insights.append("Data or modeling warnings may affect reliability and should be reviewed.")
    return insights


def _recommendations(top_features: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    recommendations = [
        "Use this report as a demo validation result, not as a production decision system.",
        "Render the chart-ready JSON in the WrenAI frontend to make model behavior easier to inspect.",
    ]
    if top_features:
        recommendations.append(f"Investigate operational drivers related to {_feature_text(top_features[:3])}.")
    if warnings:
        recommendations.append("Address warnings before expanding this workflow to real WrenAI query results.")
    return recommendations


def _limitations() -> List[str]:
    return [
        "The current demo may use sample or synthetic records.",
        "Metrics are calculated on a validation split and should not be interpreted as production performance.",
        "LLM InsightAgent is optional; rule-based report synthesis remains the fallback.",
        "The service returns chart-ready JSON, but visual chart rendering must be implemented in the frontend.",
    ]


def _feature_text(top_features: List[Dict[str, Any]]) -> str:
    return ", ".join(item["feature"] for item in top_features[:3])
