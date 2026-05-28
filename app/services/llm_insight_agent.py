from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_TIMEOUT_SECONDS = 30
SYSTEM_PROMPT = (
    "You are a professional data analyst generating a concise insight report "
    "from model evaluation results, dataset profile, feature importance, and chart summaries.\n"
    "Use only the provided compact summary.\n"
    "Focus on patterns, feature signals, model behavior, and practical interpretation.\n"
    "Be concise, evidence-based, and presentation-ready.\n"
    "Do not invent metrics, columns, datasets, or causal claims.\n"
    "Do not mention raw data that is not provided.\n"
    "Do not mention implementation details, frontend, chart rendering, API, JSON, or pipeline internals.\n"
    "Do not mention synthetic records unless the compact summary explicitly says the dataset is synthetic.\n"
    "Do not repeatedly describe the work as a demo in the insight body.\n"
    "Do not include chain-of-thought or hidden reasoning.\n"
    "Return JSON only. Do not include markdown fences.\n"
    "Start with { and end with }."
)

GENERAL_REJECT_TERMS = (
    "i cannot",
    "as an ai language model",
    "json",
    "frontend",
    "api endpoint",
    "implementation",
)
HEART_DISEASE_REJECT_TERMS = (
    "good review",
    "bad review",
    "customer",
    "order",
    "delivery",
    "freight",
    "ecommerce",
    "e-commerce",
    "synthetic records",
    "implement chart",
    "frontend",
)

InsightResult = Tuple[Optional[Dict[str, Any]], Optional[str]]


def load_local_env() -> None:
    """Load a local .env file without overriding existing environment values."""

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value.startswith(("'", '"'))
        ):
            value = value[1:-1]

        os.environ[key] = value


def is_llm_enabled() -> bool:
    load_local_env()
    return os.getenv("LLM_INSIGHT_ENABLED", "").strip().lower() == "true"


def is_llm_debug_enabled() -> bool:
    load_local_env()
    return os.getenv("LLM_DEBUG", "").strip().lower() == "true"


def build_compact_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the only data payload allowed to leave the service for LLM insight."""

    model_audit = report.get("model_audit", {})
    prediction_results = report.get("prediction_results", {})
    feature_chart = _find_chart(model_audit.get("charts", []), "feature_importance")
    if not feature_chart:
        feature_chart = _find_chart(
            prediction_results.get("charts", []),
            "heart_feature_importance_ranking",
        )
    target_metadata = report.get("target_metadata", {})
    return {
        "domain_name": target_metadata.get("domain_name"),
        "target_column": target_metadata.get("target_column"),
        "positive_label": target_metadata.get("positive_label"),
        "negative_label": target_metadata.get("negative_label"),
        "task_type": target_metadata.get("task_type"),
        "dataset_overview": report.get("dataset_overview", {}),
        "eda_summary": report.get("eda", {}).get("summary", []),
        "metrics": model_audit.get("metrics", {}),
        "top_features": (feature_chart or {}).get("data", [])[:10],
        "warnings": report.get("warnings", []),
        "limitations": report.get("limitations", []),
    }


def build_fallback_agent_insights(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build a domain-safe analyst fallback when LLM output is unavailable or rejected."""

    target_metadata = report.get("target_metadata", {})
    domain_name = str(target_metadata.get("domain_name") or "Prediction")
    overview = report.get("dataset_overview", {})
    rows = overview.get("rows", 0)
    columns = overview.get("columns", 0)
    target_column = target_metadata.get("target_column") or overview.get("target") or "target"
    class_distribution = overview.get("class_distribution", {}) or {}
    model_audit = report.get("model_audit", {})
    metrics = model_audit.get("metrics", {}) or {}
    prediction_results = report.get("prediction_results", {})
    feature_chart = _find_chart(model_audit.get("charts", []), "feature_importance")
    if not feature_chart:
        feature_chart = _find_chart(
            prediction_results.get("charts", []),
            "heart_feature_importance_ranking",
        )
    top_features = (feature_chart or {}).get("data", [])[:5]

    if _is_heart_disease(domain_name, target_column):
        negative_label = target_metadata.get("negative_label", "No heart disease")
        positive_label = target_metadata.get("positive_label", "Heart disease")
        negative_count = class_distribution.get(negative_label, 0)
        positive_count = class_distribution.get(positive_label, 0)
        summary = (
            f"The analysis covers {rows} records with {columns} columns and uses "
            f"{target_column} as the classification target. The class distribution shows "
            f"{negative_count} {negative_label} cases and {positive_count} {positive_label} cases."
        )
        findings = [
            _metric_sentence(metrics, "The validation results summarize how consistently the model separates the two heart disease classes."),
            _feature_sentence(
                top_features,
                "The strongest statistical signals in the model are",
                "Feature importance was not available for this run.",
            ),
            "The model identifies statistical associations in the prepared dataset rather than clinical causation.",
        ]
        recommendations = [
            "Review the strongest features with domain experts to confirm whether the learned patterns are plausible.",
            "Validate model behavior on an external holdout dataset before using the result outside analysis.",
            "Assess calibration, threshold behavior, and subgroup performance before any clinical use.",
        ]
        risk_notes = [
            "Before clinical use, this model would require external validation, calibration, subgroup bias assessment, and expert review.",
        ]
    else:
        summary = (
            f"The analysis covers {rows} records with {columns} columns and uses "
            f"{target_column} as the prediction target."
        )
        if class_distribution:
            summary += f" The observed target distribution is {_distribution_text(class_distribution)}."
        findings = [
            _metric_sentence(metrics, "The validation metrics provide a directional view of model quality for review prediction."),
            _feature_sentence(
                top_features,
                "The strongest signals for the prediction are",
                "Feature importance was not available for this run.",
            ),
            "The output is most useful for identifying review-quality patterns that merit deeper operational analysis.",
        ]
        recommendations = [
            "Review the strongest features to identify operational levers associated with review quality.",
            "Compare model findings against recent customer feedback and fulfillment performance trends.",
            "Validate the model on a fresh holdout period before using it for ongoing decision support.",
        ]
        risk_notes = [
            "Monitor data drift, class balance, and metric stability before relying on the model for repeated decisions.",
        ]

    return {
        "enabled": False,
        "provider": "rule-based",
        "model": "domain-safe-fallback",
        "summary": summary,
        "business_insights": [item for item in findings if item],
        "recommendations": recommendations,
        "risk_notes": risk_notes,
    }


def generate_llm_insights(report: Dict[str, Any]) -> InsightResult:
    """Generate optional LLM insight and return a safe failure reason on fallback."""

    load_local_env()
    if not is_llm_enabled():
        return None, "disabled"

    api_key = _get_api_key()
    if not api_key:
        return None, "missing_api_key"

    provider = os.getenv("LLM_PROVIDER", "deepinfra").strip() or "deepinfra"
    api_base = (os.getenv("LLM_API_BASE", DEFAULT_API_BASE).strip() or DEFAULT_API_BASE).rstrip("/")
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    timeout = _get_timeout()

    compact_summary = build_compact_summary(report)
    user_prompt = _build_user_prompt(compact_summary)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 800,
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
        return None, "invalid_response_json"
    except (urllib.error.URLError, OSError):
        return None, "request_error"

    content = _extract_message_content(response_payload)
    if not content:
        return None, "missing_content"

    parsed = parse_llm_json(content)
    if not parsed:
        return None, "invalid_json"

    normalized = _normalize_insights(parsed)
    if not normalized:
        return None, "invalid_shape"

    if _violates_quality_guardrails(normalized, compact_summary):
        return None, "quality_guardrail"

    return {
        "enabled": True,
        "provider": provider,
        "model": model,
        **normalized,
    }, None


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_think_blocks(strip_markdown_fences(content))

    for candidate in (cleaned, extract_first_json_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(strip_markdown_fences(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_user_prompt(compact_summary: Dict[str, Any]) -> str:
    domain_name = str(compact_summary.get("domain_name") or "")
    domain_section = (
        "Domain guidance for Heart Disease Classification:\n"
        "- Use heart disease risk pattern, classification outcome, or statistical association language.\n"
        "- Do not use customer, order, review, business, delivery, or freight language.\n"
        "- Mention top features, class distribution, and validation metrics only when present in the compact summary.\n"
        "- Do not provide medical diagnosis, treatment advice, medication advice, or patient-care instructions.\n"
        "- Keep clinical caution concise and place it only in risk_notes.\n"
        if _is_heart_disease(domain_name, compact_summary.get("target_column"))
        else
        "Domain guidance for E-commerce Good Review:\n"
        "- Use review, customer, and order language only when relevant to the provided summary.\n"
        "- Good review and Bad review labels are allowed.\n"
        "- Recommendations should be practical business or analytical next steps.\n"
    )
    return (
        "Return one JSON object with exactly these top-level keys: "
        "summary, business_insights, recommendations, risk_notes. "
        "Do not include markdown fences, chain-of-thought, or explanatory text. "
        "Start with { and end with }.\n"
        "Semantics:\n"
        "- summary: one concise paragraph interpreting the result.\n"
        "- business_insights: data/model findings supported by metrics, class distribution, or feature importance.\n"
        "- recommendations: analytical next steps or practical actions.\n"
        "- risk_notes: concise data/model risk cautions.\n"
        f"{domain_section}"
        "Use only this compact summary:\n"
        f"{json.dumps(compact_summary, ensure_ascii=False)}"
    )


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


def _normalize_insights(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None

    summary = parsed.get("summary", "")
    if summary is None:
        summary = ""

    normalized = {
        "summary": str(summary).strip(),
        "business_insights": _string_list(parsed.get("business_insights")),
        "recommendations": _string_list(parsed.get("recommendations")),
        "risk_notes": _string_list(parsed.get("risk_notes")),
    }
    if not normalized["summary"]:
        return None
    return normalized


def _violates_quality_guardrails(
    insights: Dict[str, Any],
    compact_summary: Dict[str, Any],
) -> bool:
    text = json.dumps(insights, ensure_ascii=False).lower()
    if any(term in text for term in GENERAL_REJECT_TERMS):
        return True

    if _is_heart_disease(compact_summary.get("domain_name"), compact_summary.get("target_column")):
        return any(term in text for term in HEART_DISEASE_REJECT_TERMS)

    return False


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


def _find_chart(charts: list[Dict[str, Any]], chart_id: str) -> Optional[Dict[str, Any]]:
    for chart in charts:
        if chart.get("id") == chart_id:
            return chart
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_heart_disease(domain_name: Any, target_column: Any) -> bool:
    domain_text = str(domain_name or "").lower()
    target_text = str(target_column or "").lower()
    return "heart" in domain_text or target_text == "heartdisease"


def _metric_sentence(metrics: Dict[str, Any], fallback: str) -> str:
    if not metrics:
        return fallback
    ordered_keys = ["accuracy", "precision", "recall", "f1", "mae", "rmse", "r2"]
    metric_parts = []
    for key in ordered_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            metric_parts.append(f"{key}={float(value):.3f}")
    if not metric_parts:
        return fallback
    return f"Validation metrics are {', '.join(metric_parts[:4])}."


def _feature_sentence(top_features: list[Dict[str, Any]], opening: str, fallback: str) -> str:
    feature_names = [str(item.get("feature")) for item in top_features[:3] if item.get("feature")]
    if not feature_names:
        return fallback
    return f"{opening} {', '.join(feature_names)}."


def _distribution_text(distribution: Dict[str, Any]) -> str:
    return ", ".join(f"{label}: {value}" for label, value in distribution.items())
