from __future__ import annotations

from typing import Any, Dict

from app.agents.base import BaseAgent
from app.schemas import TargetTransform
from app.services.report import build_prediction_report, render_report_markdown


class ReportAgent(BaseAgent):
    name = "ReportAgent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request = context["request"]
        report = build_prediction_report(
            title=self._report_title(context["task_label"]),
            task_label=context["task_label"],
            target_definition=self._target_definition(
                context["target_name"], request.target_transform
            ),
            dataset_summary=context["dataset_summary"],
            target_summary=context["target_summary"],
            model=context["model_info"],
            metrics=context["modeling"].metrics,
            charts=context["charts"],
            top_features=context["modeling"].top_features,
            sample_predictions=context["modeling"].sample_predictions,
            warnings=context["warnings"],
        )
        context["report"] = report
        self.add_trace(context, "success", "Built structured prediction report.")
        return context

    def refresh_markdown(self, report: Dict[str, Any]) -> None:
        report["report_markdown"] = render_report_markdown(report)

    def _report_title(self, task_label: str) -> str:
        if task_label == "Good review prediction":
            return "AutoMind E-commerce Good Review Prediction Report"
        if task_label == "Heart disease classification":
            return "AutoMind Heart Disease Classification Report"
        return f"AutoMind {task_label} Report"

    def _target_definition(self, target_name: str, transform: TargetTransform | None) -> str:
        if target_name == "good_review":
            return "good_review = 1 if review_score >= 4 else 0"
        if target_name == "HeartDisease":
            return "HeartDisease = 1 indicates heart disease; 0 indicates no heart disease"
        if transform and transform.type == "binary_threshold":
            return f"{target_name} is derived using {transform.operator} {transform.threshold}"
        return f"Target column: {target_name}"
