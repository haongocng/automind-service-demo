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
    "You are a lightweight business insight agent for prediction reports.\n"
    "Use only the provided compact summary.\n"
    "Do not invent numbers, columns, datasets, or causal claims.\n"
    "Do not mention raw data that is not provided.\n"
    "Do not include chain-of-thought or hidden reasoning.\n"
    "Return JSON only. Do not include markdown fences.\n"
    "Start with { and end with }."
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
    feature_chart = _find_chart(model_audit.get("charts", []), "feature_importance")
    return {
        "dataset_overview": report.get("dataset_overview", {}),
        "eda_summary": report.get("eda", {}).get("summary", []),
        "metrics": model_audit.get("metrics", {}),
        "top_features": (feature_chart or {}).get("data", [])[:10],
        "warnings": report.get("warnings", []),
        "limitations": report.get("limitations", []),
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
    user_prompt = (
        "Return one JSON object with exactly these top-level keys: "
        "summary, business_insights, recommendations, risk_notes. "
        "Do not include markdown fences, chain-of-thought, or explanatory text. "
        "Start with { and end with }. "
        "Use only this compact summary:\n"
        f"{json.dumps(compact_summary, ensure_ascii=False)}"
    )
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

    return {
        "summary": str(summary).strip(),
        "business_insights": _string_list(parsed.get("business_insights")),
        "recommendations": _string_list(parsed.get("recommendations")),
        "risk_notes": _string_list(parsed.get("risk_notes")),
    }


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
