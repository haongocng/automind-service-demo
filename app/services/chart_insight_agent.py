from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.services.llm_insight_agent import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    is_llm_enabled,
    load_local_env,
    parse_llm_json,
)


AnalysisResult = Tuple[str, Optional[str]]
HEART_TARGET = "HeartDisease"
SYSTEM_PROMPT = (
    "You are a professional data analyst writing chart-level explanations for a prediction report.\n"
    "Use only the compact chart summary provided.\n"
    "Cite concrete values, feature names, class names, ranks, counts, and percentages when available.\n"
    "For correlations, mention direction and magnitude without claiming causality.\n"
    "Compare groups/classes when the chart summary supports it.\n"
    "Explain what each chart implies for the classification task.\n"
    "Avoid vague comments such as 'this chart provides insights'.\n"
    "Avoid implementation details, frontend, API, JSON, or pipeline wording.\n"
    "For Heart Disease, discuss statistical associations, classification outcomes, model signals, validation split behavior, and risk patterns only.\n"
    "Do not provide diagnosis, treatment advice, medication advice, or patient-care instructions.\n"
    "Do not use E-commerce language or Good review / Bad review wording.\n"
    "Return only the requested object. Do not include markdown fences."
)
FORBIDDEN_HEART_TERMS = (
    "good review",
    "bad review",
    "customer",
    "order",
    "delivery",
    "freight",
    "ecommerce",
    "e-commerce",
    "diagnosis recommendation",
    "treatment recommendation",
    "medication",
    "frontend",
    "api endpoint",
    "pipeline",
    "implementation",
    "json",
)
GENERIC_PHRASES = (
    "this chart provides insights",
    "provides insights",
    "review with domain experts",
    "consult domain experts",
)


def apply_chart_insights(report: Dict[str, Any]) -> AnalysisResult:
    """Attach optional chart-level analysis to supported report charts."""

    target_metadata = report.get("target_metadata", {})
    if target_metadata.get("target_column") != HEART_TARGET:
        return "skipped", None

    fallback = build_fallback_chart_insights(report)
    llm_insights, failure_reason = generate_chart_insights(report)
    insights = llm_insights or fallback

    _attach_chart_insights(report, insights)
    return ("llm" if llm_insights else "fallback"), failure_reason


def build_chart_summary(
    report: Dict[str, Any],
    domain_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    domain_metadata = domain_metadata or report.get("target_metadata", {})
    return {
        "domain_name": domain_metadata.get("domain_name"),
        "target_column": domain_metadata.get("target_column"),
        "positive_label": domain_metadata.get("positive_label"),
        "negative_label": domain_metadata.get("negative_label"),
        "charts": [
            _compact_chart(chart, _prediction_reference_chart(report))
            for chart in _all_report_charts(report)
            if _is_supported_chart(chart)
        ],
        "metrics": report.get("model_audit", {}).get("metrics", {}),
        "limitations": report.get("limitations", []),
    }


def generate_chart_insights(report: Dict[str, Any]) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[str]]:
    load_local_env()
    if not is_llm_enabled():
        return None, "disabled"

    api_key = _get_api_key()
    if not api_key:
        return None, "missing_key"

    compact_summary = build_chart_summary(report)
    if not compact_summary.get("charts"):
        return None, "no_charts"

    api_base = (os.getenv("LLM_API_BASE", DEFAULT_API_BASE).strip() or DEFAULT_API_BASE).rstrip("/")
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    timeout = _get_timeout()
    user_prompt = (
        "Return one object with a top-level chart_insights mapping. "
        "Each key must be a chart id from the input. Each value must contain: "
        "headline, what_it_shows, key_observations, interpretation, caveat. "
        "Use concise report-ready wording. Use only this compact chart summary:\n"
        f"{json.dumps(compact_summary, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 1200,
    }

    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, f"http_{exc.code}"
    except TimeoutError:
        return None, "timeout"
    except json.JSONDecodeError:
        return None, "invalid_response"
    except (urllib.error.URLError, OSError):
        return None, "request_error"

    content = _extract_message_content(response_payload)
    parsed = parse_llm_json(content) if content else None
    if not parsed:
        return None, "invalid_response"

    insights = _normalize_chart_insights(parsed.get("chart_insights"))
    if not insights:
        return None, "invalid_shape"
    required_chart_ids = {chart.get("id") for chart in compact_summary.get("charts", []) if chart.get("id")}
    if required_chart_ids and not required_chart_ids.issubset(set(insights)):
        return None, "incomplete"
    if _violates_heart_guardrails(insights):
        return None, "guardrail"
    if _has_generic_chart_insights(insights, compact_summary):
        return None, "generic"

    return insights, None


def build_fallback_chart_insights(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    insights: Dict[str, Dict[str, Any]] = {}
    chart_map = {chart.get("id"): chart for chart in _all_report_charts(report)}

    feature_chart = chart_map.get("heart_feature_importance_ranking")
    if feature_chart:
        insights["heart_feature_importance_ranking"] = _feature_importance_fallback(feature_chart)

    confusion_chart = chart_map.get("heart_confusion_matrix")
    if confusion_chart:
        insights["heart_confusion_matrix"] = _confusion_matrix_fallback(confusion_chart)

    correlation_chart = chart_map.get("heart_correlation_heatmap")
    if correlation_chart:
        insights["heart_correlation_heatmap"] = _correlation_fallback(correlation_chart)

    slope_chart = chart_map.get("heart_st_slope_by_outcome")
    if slope_chart:
        insights["heart_st_slope_by_outcome"] = _st_slope_fallback(slope_chart)

    class_chart = chart_map.get("class_distribution")
    prediction_reference_chart = chart_map.get("heart_confusion_matrix") or class_chart
    if class_chart and "class_distribution" not in insights:
        insights["class_distribution"] = _class_distribution_fallback(class_chart)

    prediction_chart = chart_map.get("prediction_distribution")
    if prediction_chart and "prediction_distribution" not in insights:
        insights["prediction_distribution"] = _prediction_distribution_fallback(prediction_chart, prediction_reference_chart)

    return insights


def _feature_importance_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    rows = _top_feature_rows(chart, limit=10)
    top_features = rows[:5]
    top_names = [row["feature"] for row in top_features]
    top_text = ", ".join(top_names[:3]) if top_names else "the top-ranked features"
    observations = [
        f"Rank {row['rank']}: {row['feature']} has importance {_format_number(row['value'])}."
        for row in top_features
    ]
    dominance = _dominance_note(rows)
    if dominance:
        observations.append(dominance)

    return _analysis(
        headline=f"{top_text} are the leading model signals for the Heart disease class separation.",
        what_it_shows="This ranking lists the strongest model features by their contribution to the selected classifier.",
        key_observations=observations[:5],
        interpretation=(
            f"The model relies most on {top_text}; these features help separate Heart disease from "
            "No heart disease in the validation workflow, but they should be read as model signals rather than causes."
        ),
        caveat="Feature importance depends on the fitted model and selected validation split, so it should be checked against metrics and other charts.",
    )


def _confusion_matrix_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    counts = _confusion_counts(chart.get("data", []))
    correct = counts["true_positive"] + counts["true_negative"]
    incorrect = counts["false_positive"] + counts["false_negative"]
    total = correct + incorrect
    balance_note = ""
    if incorrect:
        diff = abs(counts["false_positive"] - counts["false_negative"])
        balance_note = (
            "The two error types are relatively balanced."
            if diff <= max(1, int(0.25 * incorrect))
            else "One error type is more common than the other and should be inspected."
        )

    return _analysis(
        headline=f"The model correctly classifies {correct} of {total} validation rows.",
        what_it_shows="This matrix compares actual classes with predicted classes on the validation split.",
        key_observations=[
            f"True positives: {counts['true_positive']} Heart disease cases were predicted as Heart disease.",
            f"True negatives: {counts['true_negative']} No heart disease cases were predicted as No heart disease.",
            f"False negatives: {counts['false_negative']} Heart disease cases were missed as No heart disease.",
            f"False positives: {counts['false_positive']} No heart disease cases were flagged as Heart disease.",
            balance_note,
        ],
        interpretation=(
            "Correct cells show where the classifier separates the two classes successfully, while off-diagonal "
            "cells show the specific validation errors that affect precision and recall."
        ),
        caveat="This audit reflects one validation split; a different split or external holdout set may shift the error counts.",
    )


def _correlation_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    notable = _notable_correlations(chart.get("data", []))
    strongest = notable[:5]
    feature_text = ", ".join(item[0] for item in strongest[:3]) if strongest else "selected numeric features"
    observations = [
        _target_correlation_sentence(feature, value)
        for feature, value in strongest
    ]
    non_target = _strongest_non_target_correlations(chart.get("data", []), limit=1)
    if non_target:
        left, right, value = non_target[0]
        observations.append(
            f"The strongest non-target numeric relationship is {left} with {right} at correlation {_format_signed(value)}."
        )

    return _analysis(
        headline=f"{feature_text} have the clearest linear association with HeartDisease.",
        what_it_shows="This heatmap summarizes pairwise correlations among numeric features and the HeartDisease target.",
        key_observations=observations[:5],
        interpretation=(
            "Positive correlations mean higher feature values appear more often with the Heart disease class; "
            "negative correlations mean higher values appear more often with the No heart disease class."
        ),
        caveat="Correlation is an association measure only and can miss non-linear patterns that the classifier may still use.",
    )


def _st_slope_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    groups = _categorical_outcome_groups(chart.get("data", []))
    observations = []
    for category in _ordered_categories(groups, ["Flat", "Up", "Down"]):
        row = groups[category]
        observations.append(
            f"{category} has {row['positive_count']} Heart disease cases and {row['negative_count']} No heart disease cases; "
            f"the dominant outcome is {row['dominant_outcome']}."
        )
    if not observations:
        observations.append("ST_Slope category counts are available but too sparse to compare reliably.")

    return _analysis(
        headline="ST_Slope categories show a clear split between the two HeartDisease outcomes.",
        what_it_shows="This grouped bar chart compares ST_Slope categories across Heart disease and No heart disease classes.",
        key_observations=observations[:5],
        interpretation=(
            "The category-level separation supports why ST_Slope-related features can rank highly in the model: "
            "some categories are concentrated in one classification outcome."
        ),
        caveat="This chart shows categorical association in the prepared dataset and should be read alongside other model signals.",
    )


def _class_distribution_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    summary = _distribution_summary(chart.get("data", []))
    positive = summary["positive_count"]
    negative = summary["negative_count"]
    total = summary["total"]
    balance = _balance_description(positive, negative)
    return _analysis(
        headline=f"The target is {balance}, with {positive} Heart disease and {negative} No heart disease rows.",
        what_it_shows="This chart shows the observed class balance in the prepared HeartDisease target.",
        key_observations=[
            f"Heart disease accounts for {_percentage(positive, total)} of the labeled rows ({positive} of {total}).",
            f"No heart disease accounts for {_percentage(negative, total)} of the labeled rows ({negative} of {total}).",
            f"The absolute class gap is {abs(positive - negative)} rows.",
        ],
        interpretation="A reasonably represented pair of classes makes accuracy, precision, recall, and F1 easier to interpret together.",
        caveat="Class balance describes the dataset, not the model's ability to separate the classes.",
    )


def _prediction_distribution_fallback(
    chart: Dict[str, Any],
    actual_chart: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    predicted = _distribution_summary(chart.get("data", []))
    actual = _actual_distribution_summary(actual_chart) if actual_chart else None
    positive = predicted["positive_count"]
    negative = predicted["negative_count"]
    total = predicted["total"]
    observations = [
        f"The model predicts Heart disease for {positive} validation rows ({_percentage(positive, total)}).",
        f"The model predicts No heart disease for {negative} validation rows ({_percentage(negative, total)}).",
    ]
    if actual and actual["total"]:
        observations.append(
            f"Observed class balance is {actual['positive_count']} Heart disease and {actual['negative_count']} No heart disease rows."
        )
        observations.append(
            f"Predicted Heart disease differs from observed Heart disease by {positive - actual['positive_count']} rows."
        )

    return _analysis(
        headline=f"Predictions lean toward {'Heart disease' if positive >= negative else 'No heart disease'} on the validation split.",
        what_it_shows="This chart shows how the classifier distributes its predicted labels across the two classes.",
        key_observations=observations[:5],
        interpretation=(
            "Prediction distribution helps reveal whether the classifier is over-concentrating on one class before reviewing the confusion matrix."
        ),
        caveat="Prediction counts do not show correctness by themselves; compare them with the confusion matrix and validation metrics.",
    )


def _compact_chart(
    chart: Dict[str, Any],
    actual_distribution_chart: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chart_id = chart.get("id")
    kind = chart.get("kind")
    compact = {
        "id": chart_id,
        "title": chart.get("title"),
        "kind": kind,
        "type": chart.get("type"),
    }
    data = list(chart.get("data") or [])
    if chart_id == "heart_correlation_heatmap" or kind == "correlation_heatmap":
        compact["correlation_summary"] = _compact_correlations(data)
    elif chart_id == "heart_feature_importance_ranking" or kind == "feature_importance":
        compact["feature_importance_summary"] = _compact_feature_importance(data)
    elif chart_id == "heart_confusion_matrix" or kind == "confusion_matrix":
        compact["confusion_matrix_summary"] = _compact_confusion_matrix(data)
    elif chart_id == "heart_st_slope_by_outcome" or kind == "categorical_outcome_distribution":
        compact["categorical_outcome_summary"] = _compact_categorical_outcome(data)
    elif chart_id == "class_distribution" or kind == "class_distribution":
        compact["class_distribution_summary"] = _compact_class_distribution(data)
    elif chart_id == "prediction_distribution" or kind == "prediction_distribution":
        compact["prediction_distribution_summary"] = _compact_prediction_distribution(data, actual_distribution_chart)
    else:
        compact["top_rows"] = data[:8]
    return compact


def _compact_feature_importance(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = _top_feature_rows({"data": data}, limit=10)
    return {
        "top_10_features": rows,
        "top_5_feature_names": [row["feature"] for row in rows[:5]],
        "dominance_note": _dominance_note(rows),
    }


def _compact_confusion_matrix(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = _confusion_counts(data)
    correct = counts["true_positive"] + counts["true_negative"]
    incorrect = counts["false_positive"] + counts["false_negative"]
    return {
        **counts,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "positive_label": "Heart disease",
        "negative_label": "No heart disease",
        "cell_explanations": {
            "true_positive": "actual Heart disease predicted as Heart disease",
            "true_negative": "actual No heart disease predicted as No heart disease",
            "false_positive": "actual No heart disease predicted as Heart disease",
            "false_negative": "actual Heart disease predicted as No heart disease",
        },
    }


def _compact_categorical_outcome(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = _categorical_outcome_groups(data)
    return {
        "groups": [groups[key] for key in _ordered_categories(groups, ["Flat", "Up", "Down"])],
        "positive_label": "Heart disease",
        "negative_label": "No heart disease",
    }


def _compact_class_distribution(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _distribution_summary(data)


def _compact_prediction_distribution(
    prediction_data: List[Dict[str, Any]],
    actual_reference_chart: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    predicted = _distribution_summary(prediction_data)
    actual = _actual_distribution_summary(actual_reference_chart) if actual_reference_chart else _distribution_summary([])
    comparison = None
    if actual["total"]:
        comparison = {
            "predicted_positive_minus_actual_positive": predicted["positive_count"] - actual["positive_count"],
            "predicted_negative_minus_actual_negative": predicted["negative_count"] - actual["negative_count"],
        }
    return {
        "predicted": predicted,
        "actual_distribution_for_comparison": actual if actual["total"] else None,
        "comparison": comparison,
    }


def _compact_correlations(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_rows = [
        {
            "feature": feature,
            "value": round(value, 3),
            "direction": "positive" if value > 0 else "negative" if value < 0 else "zero",
            "strength": _correlation_strength(abs(value)),
        }
        for feature, value in _notable_correlations(data)
    ]
    unique_pairs = _unique_correlation_pairs(data)
    positives = [row for row in unique_pairs if row["value"] > 0]
    negatives = [row for row in unique_pairs if row["value"] < 0]
    return {
        "target_correlations_sorted_by_absolute_value": target_rows,
        "top_positive_correlations": sorted(positives, key=lambda row: row["value"], reverse=True)[:5],
        "top_negative_correlations": sorted(negatives, key=lambda row: row["value"])[:5],
        "strongest_non_diagonal_correlations": sorted(
            unique_pairs,
            key=lambda row: abs(row["value"]),
            reverse=True,
        )[:8],
        "self_correlations_excluded": True,
    }


def _notable_correlations(data: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    rows: Dict[str, float] = {}
    for row in data:
        left = row.get("feature_x")
        right = row.get("feature_y")
        if HEART_TARGET not in {left, right} or left == right:
            continue
        feature = str(right if left == HEART_TARGET else left)
        try:
            value = float(row.get("value", 0))
        except (TypeError, ValueError):
            continue
        if feature not in rows or abs(value) > abs(rows[feature]):
            rows[feature] = value
    return sorted(rows.items(), key=lambda item: abs(item[1]), reverse=True)



def _top_feature_rows(chart: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    rows = []
    for index, row in enumerate(chart.get("data", []) or [], start=1):
        feature = row.get("feature")
        value = row.get("value", row.get("importance"))
        if feature is None or value is None:
            continue
        numeric_value = _safe_float(value)
        if numeric_value is None:
            continue
        rank = _safe_int(row.get("rank")) or index
        rows.append({"rank": rank, "feature": str(feature), "value": round(numeric_value, 4)})
    return sorted(rows, key=lambda item: (item["rank"], -item["value"]))[:limit]


def _dominance_note(rows: List[Dict[str, Any]]) -> str:
    if len(rows) < 2:
        return ""
    top = float(rows[0].get("value", 0))
    second = float(rows[1].get("value", 0))
    if second > 0 and top >= second * 1.75:
        return (
            f"{rows[0]['feature']} is notably dominant: its importance {_format_number(top)} "
            f"is about {_format_number(top / second)} times the second-ranked feature."
        )
    return ""


def _confusion_counts(data: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    for row in data:
        actual = _label_kind(row.get("actual"))
        predicted = _label_kind(row.get("predicted"))
        value = _safe_int(row.get("value")) or 0
        if actual == "positive" and predicted == "positive":
            counts["true_positive"] += value
        elif actual == "negative" and predicted == "negative":
            counts["true_negative"] += value
        elif actual == "negative" and predicted == "positive":
            counts["false_positive"] += value
        elif actual == "positive" and predicted == "negative":
            counts["false_negative"] += value
    return counts


def _categorical_outcome_groups(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for row in data:
        category = str(row.get("category", "Unknown"))
        outcome = _label_kind(row.get("outcome"))
        value = _safe_int(row.get("value")) or 0
        group = groups.setdefault(
            category,
            {
                "category": category,
                "positive_count": 0,
                "negative_count": 0,
                "total": 0,
                "dominant_outcome": "Tie",
            },
        )
        if outcome == "positive":
            group["positive_count"] += value
        elif outcome == "negative":
            group["negative_count"] += value
        group["total"] += value

    for group in groups.values():
        positive = group["positive_count"]
        negative = group["negative_count"]
        if positive > negative:
            group["dominant_outcome"] = "Heart disease"
        elif negative > positive:
            group["dominant_outcome"] = "No heart disease"
        else:
            group["dominant_outcome"] = "Tie"
        total = group["total"]
        group["positive_percentage"] = _percentage(positive, total)
        group["negative_percentage"] = _percentage(negative, total)
    return groups


def _ordered_categories(groups: Dict[str, Dict[str, Any]], preferred: List[str]) -> List[str]:
    ordered = [category for category in preferred if category in groups]
    remaining = sorted((category for category in groups if category not in ordered), key=lambda key: groups[key]["total"], reverse=True)
    return ordered + remaining


def _distribution_summary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    positive = 0
    negative = 0
    other = 0
    rows = []
    for row in data:
        label = str(row.get("label", ""))
        value = _safe_int(row.get("value")) or 0
        kind = _label_kind(label)
        rows.append({"label": label, "value": value})
        if kind == "positive":
            positive += value
        elif kind == "negative":
            negative += value
        else:
            other += value
    total = positive + negative + other
    return {
        "positive_label": "Heart disease",
        "negative_label": "No heart disease",
        "positive_count": positive,
        "negative_count": negative,
        "other_count": other,
        "total": total,
        "positive_percentage": _percentage(positive, total),
        "negative_percentage": _percentage(negative, total),
        "rows": rows,
    }


def _unique_correlation_pairs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    pairs = []
    for row in data:
        left = str(row.get("feature_x"))
        right = str(row.get("feature_y"))
        if left == right:
            continue
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        value = _safe_float(row.get("value"))
        if value is None:
            continue
        pairs.append({"feature_x": left, "feature_y": right, "value": round(value, 3)})
    return pairs


def _strongest_non_target_correlations(data: List[Dict[str, Any]], limit: int = 3) -> List[Tuple[str, str, float]]:
    pairs = []
    for row in _unique_correlation_pairs(data):
        left = row["feature_x"]
        right = row["feature_y"]
        if HEART_TARGET in {left, right}:
            continue
        pairs.append((left, right, float(row["value"])))
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)[:limit]


def _target_correlation_sentence(feature: str, value: float) -> str:
    direction = "positive" if value > 0 else "negative" if value < 0 else "near-zero"
    strength = _correlation_strength(abs(value))
    if value > 0:
        meaning = "higher values appear more often with the Heart disease class"
    elif value < 0:
        meaning = "higher values appear more often with the No heart disease class"
    else:
        meaning = "there is little visible linear separation by this feature"
    return f"{feature} has a {strength} {direction} correlation with HeartDisease ({_format_signed(value)}), meaning {meaning}."


def _correlation_strength(value: float) -> str:
    if value >= 0.5:
        return "strong"
    if value >= 0.3:
        return "moderate"
    if value >= 0.1:
        return "weak"
    return "very weak"


def _balance_description(positive: int, negative: int) -> str:
    total = positive + negative
    if total == 0:
        return "unavailable"
    minority = min(positive, negative)
    ratio = minority / total
    if ratio >= 0.4:
        return "well balanced"
    if ratio >= 0.25:
        return "moderately balanced"
    return "imbalanced"


def _label_kind(value: Any) -> str:
    label = str(value or "").strip().lower()
    label = label.replace("predicted ", "")
    if "no heart disease" in label:
        return "negative"
    if "heart disease" in label or label == "1":
        return "positive"
    if label == "0":
        return "negative"
    return "other"


def _format_number(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return str(value)
    if abs(numeric) >= 10:
        return f"{numeric:.1f}"
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


def _percentage(part: int, total: int) -> str:
    if not total:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_chart(report: Dict[str, Any], chart_id: str) -> Optional[Dict[str, Any]]:
    for chart in _all_report_charts(report):
        if chart.get("id") == chart_id:
            return chart
    return None



def _prediction_reference_chart(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _find_chart(report, "heart_confusion_matrix") or _find_chart(report, "class_distribution")


def _actual_distribution_summary(reference_chart: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not reference_chart:
        return _distribution_summary([])
    data = list(reference_chart.get("data") or [])
    if reference_chart.get("id") == "heart_confusion_matrix" or reference_chart.get("kind") == "confusion_matrix":
        counts = _confusion_counts(data)
        positive = counts["true_positive"] + counts["false_negative"]
        negative = counts["true_negative"] + counts["false_positive"]
        total = positive + negative
        return {
            "positive_label": "Heart disease",
            "negative_label": "No heart disease",
            "positive_count": positive,
            "negative_count": negative,
            "other_count": 0,
            "total": total,
            "positive_percentage": _percentage(positive, total),
            "negative_percentage": _percentage(negative, total),
            "rows": [
                {"label": "Heart disease", "value": positive},
                {"label": "No heart disease", "value": negative},
            ],
        }
    return _distribution_summary(data)

def _has_generic_chart_insights(
    insights: Dict[str, Dict[str, Any]],
    compact_summary: Dict[str, Any],
) -> bool:
    charts = compact_summary.get("charts", []) or []
    for chart in charts:
        chart_id = chart.get("id")
        if not chart_id:
            continue
        insight = insights.get(chart_id)
        if not insight or _is_generic_chart_insight(insight, chart):
            return True
    return False


def _is_generic_chart_insight(insight: Dict[str, Any], chart_summary: Dict[str, Any]) -> bool:
    observations = _string_list(insight.get("key_observations"))
    meaningful_observations = [item for item in observations if len(item.split()) >= 5]
    if len(meaningful_observations) < 2:
        return True

    text = json.dumps(insight, ensure_ascii=False).lower()
    if any(phrase in text for phrase in GENERIC_PHRASES):
        return True

    tokens = _specific_tokens(chart_summary)
    if tokens and not any(token.lower() in text for token in tokens):
        return True
    return False


def _specific_tokens(value: Any) -> List[str]:
    tokens = set()
    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"id", "title", "kind", "type", "direction", "strength", "self_correlations_excluded"}:
                    continue
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, (int, float)):
            tokens.add(str(round(float(item), 3)).rstrip("0").rstrip("."))
        elif isinstance(item, str):
            stripped = item.strip()
            if len(stripped) >= 2 and stripped.lower() not in {"heart disease", "no heart disease", "tie"}:
                tokens.add(stripped)
    visit(value)
    return sorted(tokens, key=len, reverse=True)[:80]

def _attach_chart_insights(report: Dict[str, Any], insights: Dict[str, Dict[str, Any]]) -> None:
    for chart in _all_report_charts(report):
        chart_id = chart.get("id")
        if chart_id in insights:
            chart["analysis"] = insights[chart_id]


def _all_report_charts(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []
    charts.extend(report.get("eda", {}).get("charts", []) or [])
    charts.extend(report.get("prediction_results", {}).get("charts", []) or [])
    charts.extend(report.get("model_audit", {}).get("charts", []) or [])
    return charts


def _is_supported_chart(chart: Dict[str, Any]) -> bool:
    return chart.get("id") in {
        "heart_feature_importance_ranking",
        "heart_confusion_matrix",
        "heart_correlation_heatmap",
        "heart_st_slope_by_outcome",
        "class_distribution",
        "prediction_distribution",
    }


def _analysis(
    headline: str,
    what_it_shows: str,
    key_observations: List[str],
    interpretation: str,
    caveat: str,
) -> Dict[str, Any]:
    return {
        "headline": headline,
        "what_it_shows": what_it_shows,
        "key_observations": [item for item in key_observations if item],
        "interpretation": interpretation,
        "caveat": caveat,
    }


def _normalize_chart_insights(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for chart_id, raw in value.items():
        if not isinstance(raw, dict):
            continue
        normalized[str(chart_id)] = _analysis(
            headline=str(raw.get("headline", "")).strip(),
            what_it_shows=str(raw.get("what_it_shows", "")).strip(),
            key_observations=_string_list(raw.get("key_observations")),
            interpretation=str(raw.get("interpretation", "")).strip(),
            caveat=str(raw.get("caveat", "")).strip(),
        )
    return {key: value for key, value in normalized.items() if value.get("headline")}


def _violates_heart_guardrails(insights: Dict[str, Dict[str, Any]]) -> bool:
    text = json.dumps(insights, ensure_ascii=False).lower()
    return any(term in text for term in FORBIDDEN_HEART_TERMS)


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_message_content(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message", {})
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message.get("content") or "")
    if first_choice.get("text") is not None:
        return str(first_choice.get("text") or "")
    return ""


def _get_api_key() -> str:
    for key in ("LLM_API_KEY", "DEEP_INFRA_API_KEY", "DEEPINFRA_API_KEY"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _get_timeout() -> int:
    try:
        return int(os.getenv("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
