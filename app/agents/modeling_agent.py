from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.agents.base import BaseAgent
from app.services.domain_metadata import format_target_label
from app.services.modeling import train_and_evaluate


class ModelingAgent(BaseAgent):
    name = "ModelingAgent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request = context["request"]
        preprocessing = context["preprocessing"]
        modeling = train_and_evaluate(
            preprocessing.X,
            context["y"],
            preprocessing.preprocessor,
            request.task_type,
            context["target_name"],
            row_metadata=context["modeling_df"],
        )
        context.setdefault("warnings", []).extend(modeling.warnings)
        charts = {
            "feature_importance": modeling.charts.get("feature_importance", []),
            "class_distribution": self._class_distribution_for_target(
                context["y"], context["target_name"]
            )
            if request.task_type == "classification"
            else [],
            "prediction_distribution": modeling.charts.get("prediction_distribution", []),
        }
        context.update({"modeling": modeling, "charts": charts})
        self.add_trace(context, "success", "Trained and evaluated candidate models.")
        return context

    def _class_distribution_for_target(self, y: pd.Series, target_name: str) -> List[Dict[str, Any]]:
        counts = y.value_counts().sort_index()
        items: List[Dict[str, Any]] = []
        for label, count in counts.items():
            items.append(
                {
                    "label": format_target_label(label, target_name),
                    "value": int(count),
                }
            )
        return items
