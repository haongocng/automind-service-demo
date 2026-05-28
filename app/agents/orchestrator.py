from __future__ import annotations

from typing import Any, Dict, List

from app.agents.data_profiler_agent import DataProfilerAgent
from app.agents.insight_agent import InsightAgent
from app.agents.model_audit_agent import ModelAuditAgent
from app.agents.modeling_agent import ModelingAgent
from app.agents.preprocessing_agent import PreprocessingAgent
from app.agents.report_agent import ReportAgent
from app.schemas import PredictionRequest
from app.services.insight import generate_insight


class AutoMindOrchestrator:
    """Lightweight orchestrator that preserves the existing response contract."""

    def __init__(self) -> None:
        self.data_profiler = DataProfilerAgent()
        self.preprocessing = PreprocessingAgent()
        self.modeling = ModelingAgent()
        self.model_audit = ModelAuditAgent()
        self.report = ReportAgent()
        self.insight = InsightAgent()

    def run(
        self,
        request: PredictionRequest,
        task_label: str = "prediction",
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "request": request,
            "task_label": task_label,
            "warnings": [],
            "agent_trace": [],
        }

        self.data_profiler.run(context)
        if context["df"].empty:
            return self._warning_response(
                task_label,
                context["dataset_summary"],
                context["warnings"],
                context["agent_trace"],
            )

        self.preprocessing.run(context)
        self.modeling.run(context)
        self.model_audit.run(context)
        self.report.run(context)
        self.insight.run(context)

        modeling = context["modeling"]
        status = "success" if modeling.metrics else "warning"
        summary = {
            "task": task_label,
            "rows": int(len(context["modeling_df"])),
            "target": context["target_name"],
            "selected_model": modeling.selected_model,
        }
        legacy = {
            "summary": summary,
            "metrics": modeling.metrics,
            "charts": context["charts"],
            "insight": context["insight"],
            "warnings": context["warnings"],
        }
        context["report"]["agent_workflow"] = context["agent_trace"]
        self.report.refresh_markdown(context["report"])

        return {
            "status": status,
            "report": context["report"],
            "legacy": legacy,
            "summary": summary,
            "dataset_summary": context["dataset_summary"],
            "target_summary": context["target_summary"],
            "model": context["model_info"],
            "metrics": modeling.metrics,
            "charts": context["charts"],
            "top_features": modeling.top_features,
            "insight": context["insight"],
            "warnings": context["warnings"],
            "agent_trace": context["agent_trace"],
        }

    def _warning_response(
        self,
        task_label: str,
        dataset_summary: Dict[str, Any],
        warnings: List[str],
        agent_trace: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        return {
            "status": "warning",
            "summary": {
                "task": task_label,
                "rows": dataset_summary.get("rows", 0),
                "target": None,
                "selected_model": None,
            },
            "dataset_summary": dataset_summary,
            "target_summary": {},
            "model": {"selected_model": None, "candidate_models": [], "dropped_columns": []},
            "metrics": {},
            "charts": {
                "feature_importance": [],
                "class_distribution": [],
                "prediction_distribution": [],
            },
            "top_features": [],
            "insight": generate_insight(task_label, {}, [], warnings),
            "warnings": sorted(set(warnings)),
            "agent_trace": agent_trace,
        }
