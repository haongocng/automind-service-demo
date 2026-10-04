"""Descriptive analysis of full-population SQL evidence, without an LLM or training.

The Wren adapter freezes the data and computes aggregates. No preview records or
fabricated findings are used. This is a descriptive task alongside prediction,
not an implementation of the paper's complete autonomous research algorithm.
"""
from typing import Any
from pydantic import BaseModel, Field


class ExplorationRequest(BaseModel):
    snapshotId: str = Field(min_length=1, max_length=100)
    goal: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any]
    profile: dict[str, Any]
    evidence: list[dict[str, Any]] = Field(max_length=10)
    method: str


def analyze_exploration(request: ExplorationRequest) -> dict[str, Any]:
    profile = request.profile
    context = request.context
    count = int(profile["rowCount"])
    if count < 0:
        raise ValueError("Row count cannot be negative")
    summary = [
        f"Analyzed {count:,} rows in {context['datasetName']} / {context['viewName']} "
        f"with the saved filters. Each row represents: {context['rowGrain'].lower()}."
    ]
    if profile.get("dateMin") and profile.get("dateMax"):
        summary.append(f"Filtered date coverage: {str(profile['dateMin'])[:10]} to {str(profile['dateMax'])[:10]}.")
    metrics = profile.get("metrics")
    if metrics:
        amount = "Not available" if metrics["itemSales"] is None else f"{metrics['itemSales']:,.2f}"
        summary.append(f"Total item sales: {amount} price units, excluding freight. Distinct orders: {metrics['distinctOrders']:,}.")
    if count == 0:
        summary.append("No rows match this scope. No trend or ranking can be inferred.")
    evidence = []
    for item in request.evidence:
        values = item.get("values", [])
        valid = [r for r in values if r.get("value") is not None]
        if valid and count:
            largest = max(valid, key=lambda r: r["value"])
            chart = item.get("chart", {})
            verb = "Peak period" if chart.get("kind") == "time" else "Largest displayed group"
            summary.append(
                f"{item['title']}: {verb.lower()} is {largest['label']} "
                f"with {largest['value']:,.2f} ({item['metricLabel']}; {item['scope'].lower()})."
            )
        evidence.append({"title": item["title"], "definition": item["definition"] + "; " + item["scope"], "values": values})
    missing = [f for f in profile.get("fields", []) if f.get("nullCount", 0) > 0]
    if missing:
        summary.append("Missing values occur in: " + ", ".join(f"{f['name']} ({f['nullCount']:,})" for f in missing[:5]) + ".")
    checks = ["Inspect missing values and confirm field meanings before making operational decisions."]
    if profile.get("dateMin"):
        checks.insert(0, "Compare equal-length date periods before interpreting a trend; partial periods can distort comparisons.")
    if context.get("viewId") == "sales":
        checks.insert(0, "Compare item sales and distinct order counts together. Freight is excluded and currency is unspecified.")
    if not count:
        checks.insert(0, "Broaden the filters or inspect the original data view, then run a new analysis.")
    return {
        "summary": summary,
        "evidence": evidence,
        "nextChecks": checks,
        "technical": {
            "method": "Descriptive SQL analysis",
            "scopeMethod": request.method,
            "snapshotId": request.snapshotId,
            "datasetVersion": profile.get("datasetVersion"),
            "schemaVersion": context.get("schemaVersion"),
            "goal": request.goal,
            "metricDefinitions": context.get("metricDefinitions", []),
            "queries": [item.get("sql") for item in request.evidence],
            "limitations": [
                "Findings describe observed aggregates; they do not establish causation or predict future outcomes.",
                "The goal is recorded as context. This task does not translate arbitrary goals into new SQL or use an LLM.",
                "Top-category evidence contains only the configured groups; distinct order counts are not additive across arbitrary groups.",
            ],
        },
    }
