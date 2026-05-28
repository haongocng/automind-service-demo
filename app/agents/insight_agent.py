from __future__ import annotations

from typing import Any, Dict

from app.agents.base import BaseAgent
from app.services.insight import generate_insight
from app.services.llm_insight_agent import (
    generate_llm_insights,
    is_llm_debug_enabled,
    is_llm_enabled,
)
from app.services.report import render_report_markdown


class InsightAgent(BaseAgent):
    name = "InsightAgent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        modeling = context["modeling"]
        insight = generate_insight(
            context["task_label"],
            modeling.metrics,
            modeling.top_features,
            context["warnings"],
        )
        context["insight"] = insight

        llm_status = "disabled"
        llm_requested = is_llm_enabled()
        llm_insights, llm_failure_reason = generate_llm_insights(context["report"])
        if llm_insights:
            context["report"]["agent_insights"] = llm_insights
            context["report"]["report_markdown"] = render_report_markdown(context["report"])
            llm_status = "success"
        elif llm_requested:
            llm_status = "fallback"
            if is_llm_debug_enabled() and llm_failure_reason:
                llm_warning = (
                    f"LLM InsightAgent failed: {llm_failure_reason}; "
                    "rule-based insights were used."
                )
            else:
                llm_warning = "LLM InsightAgent failed; rule-based insights were used."
            context["warnings"] = sorted(set(context["warnings"] + [llm_warning]))
            context["report"]["warnings"] = context["warnings"]
            context["report"]["report_markdown"] = render_report_markdown(context["report"])

        self.add_trace(
            context,
            "success",
            f"Generated rule-based insight; llm={llm_status}.",
        )
        return context
