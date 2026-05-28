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
    "You are a professional data analyst writing concise chart-level explanations.\n"
    "Use only the compact chart summary provided.\n"
    "Explain what each chart shows, the most important pattern, why it matters, and one caveat.\n"
    "For Heart Disease, discuss statistical patterns and model behavior only.\n"
    "Do not provide diagnosis, treatment advice, medication advice, or patient-care instructions.\n"
    "Do not mention implementation details, frontend, API, or pipeline internals.\n"
    "Do not use E-commerce language.\n"
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
    "implementation",
    "json",
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
        "charts": [_compact_chart(chart) for chart in _all_report_charts(report) if _is_supported_chart(chart)],
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
    if _violates_heart_guardrails(insights):
        return None, "guardrail"

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
    if class_chart and "class_distribution" not in insights:
        insights["class_distribution"] = _class_distribution_fallback(class_chart)

    prediction_chart = chart_map.get("prediction_distribution")
    if prediction_chart and "prediction_distribution" not in insights:
        insights["prediction_distribution"] = _prediction_distribution_fallback(prediction_chart)

    return insights


def _feature_importance_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    rows = chart.get("data", [])[:3]
    features = [str(row.get("feature")) for row in rows if row.get("feature")]
    top_text = ", ".join(features) if features else "the top-ranked features"
    return _analysis(
        headline=f"{top_text} are the strongest model signals.",
        what_it_shows="This chart ranks features by their contribution to the selected classification model.",
        key_observations=[
            f"The highest-ranked features are {top_text}.",
            "Higher bars indicate greater influence in the fitted model.",
        ],
        interpretation="Higher-ranked features have stronger influence on model predictions, but they do not imply clinical causation.",
        caveat="Feature importance should be reviewed with validation metrics and domain expertise.",
    )


def _confusion_matrix_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    rows = chart.get("data", [])
    correct = sum(int(row.get("value", 0)) for row in rows if row.get("actual") == row.get("predicted"))
    incorrect = sum(int(row.get("value", 0)) for row in rows if row.get("actual") != row.get("predicted"))
    return _analysis(
        headline="The validation split shows more correct than incorrect predictions in both classes.",
        what_it_shows="This matrix compares actual versus predicted classes on the validation split.",
        key_observations=[
            f"Correct predictions total {correct} validation rows.",
            f"Off-diagonal cells account for {incorrect} misclassified validation rows.",
        ],
        interpretation="The off-diagonal cells identify where the classifier confuses the two classes and should guide model audit.",
        caveat="This matrix is based on one validation split and should be checked on external holdout data.",
    )


def _correlation_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    notable = _notable_correlations(chart.get("data", []))
    feature_names = [item[0] for item in notable[:3]]
    feature_text = ", ".join(feature_names) if feature_names else "selected numeric features"
    return _analysis(
        headline=f"{feature_text} show visible relationships with the HeartDisease target.",
        what_it_shows="This heatmap summarizes pairwise correlations among numeric features and the target.",
        key_observations=[
            *(f"{feature} has correlation {value:.3f} with HeartDisease." for feature, value in notable[:3]),
        ] or ["The chart highlights linear associations among numeric fields."],
        interpretation="Correlation highlights linear associations and should be interpreted together with model-based feature importance.",
        caveat="Correlation does not establish causation and may miss non-linear relationships.",
    )


def _st_slope_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    rows = chart.get("data", [])
    by_category: Dict[str, Dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category"))
        outcome = str(row.get("outcome"))
        by_category.setdefault(category, {})[outcome] = int(row.get("value", 0))

    flat = by_category.get("Flat", {})
    up = by_category.get("Up", {})
    return _analysis(
        headline="ST_Slope categories separate the two outcome groups strongly.",
        what_it_shows="This chart compares ST_Slope categories across HeartDisease classes.",
        key_observations=[
            f"Flat includes {flat.get('Heart disease', 0)} Heart disease cases and {flat.get('No heart disease', 0)} No heart disease cases.",
            f"Up includes {up.get('No heart disease', 0)} No heart disease cases and {up.get('Heart disease', 0)} Heart disease cases.",
        ],
        interpretation="Flat is concentrated among Heart disease cases, while Up is more common in No heart disease cases.",
        caveat="This is a categorical association and should be considered alongside other features and validation results.",
    )


def _class_distribution_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    rows = chart.get("data", [])
    summary = ", ".join(f"{row.get('label')}: {row.get('value')}" for row in rows)
    return _analysis(
        headline="The target classes are both represented in the prepared dataset.",
        what_it_shows="This chart shows the number of labeled rows in each HeartDisease class.",
        key_observations=[summary] if summary else ["The chart summarizes class counts."],
        interpretation="Class balance affects how validation metrics should be interpreted.",
        caveat="Class counts alone do not indicate model quality.",
    )


def _prediction_distribution_fallback(chart: Dict[str, Any]) -> Dict[str, Any]:
    rows = chart.get("data", [])
    summary = ", ".join(f"{row.get('label')}: {row.get('value')}" for row in rows)
    return _analysis(
        headline="Predicted classes are distributed across both outcomes.",
        what_it_shows="This chart shows how the selected model distributed validation predictions.",
        key_observations=[summary] if summary else ["The chart summarizes predicted class counts."],
        interpretation="Prediction distribution helps identify whether the model is over-concentrating on one class.",
        caveat="Prediction counts should be interpreted together with the confusion matrix.",
    )


def _compact_chart(chart: Dict[str, Any]) -> Dict[str, Any]:
    chart_id = chart.get("id")
    compact = {
        "id": chart_id,
        "title": chart.get("title"),
        "kind": chart.get("kind"),
        "type": chart.get("type"),
    }
    data = list(chart.get("data") or [])
    if chart_id == "heart_correlation_heatmap":
        compact["notable_correlations"] = _compact_correlations(data)
    elif chart_id == "heart_confusion_matrix":
        compact["matrix"] = data[:6]
    elif chart_id == "heart_st_slope_by_outcome":
        compact["group_counts"] = data[:12]
    else:
        compact["top_rows"] = data[:8]
    return compact


def _compact_correlations(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target_rows = [
        row
        for row in data
        if HEART_TARGET in {row.get("feature_x"), row.get("feature_y")}
        and row.get("feature_x") != row.get("feature_y")
    ]
    strongest_rows = [
        row
        for row in data
        if row.get("feature_x") != row.get("feature_y")
    ]
    combined = target_rows + sorted(
        strongest_rows,
        key=lambda row: abs(float(row.get("value", 0))),
        reverse=True,
    )
    seen = set()
    output = []
    for row in combined:
        key = (row.get("feature_x"), row.get("feature_y"))
        reverse_key = (row.get("feature_y"), row.get("feature_x"))
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        output.append(row)
        if len(output) >= 10:
            break
    return output


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
