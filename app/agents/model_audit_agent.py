from __future__ import annotations

from typing import Any, Dict

from app.agents.base import BaseAgent


class ModelAuditAgent(BaseAgent):
    name = "ModelAuditAgent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        modeling = context["modeling"]
        preprocessing = context["preprocessing"]
        request = context["request"]
        target_summary = {**context["target_summary"], "task_type": request.task_type}
        model_info = {
            "selected_model": modeling.selected_model,
            "candidate_models": modeling.candidate_models,
            "dropped_columns": preprocessing.dropped_columns,
            "numeric_features": preprocessing.numeric_features,
            "categorical_features": preprocessing.categorical_features,
        }
        context["target_summary"] = target_summary
        context["model_info"] = model_info
        context["warnings"] = sorted(set(context.get("warnings", [])))
        self.add_trace(context, "success", "Organized model audit outputs.")
        return context
