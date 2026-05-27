from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.schemas import PredictionRequest, TargetTransform
from app.services.insight import generate_insight
from app.services.llm_insight_agent import (
    generate_llm_insights,
    is_llm_debug_enabled,
    is_llm_enabled,
)
from app.services.modeling import train_and_evaluate
from app.services.preprocessing import prepare_features
from app.services.profiler import profile_dataframe
from app.services.report import build_prediction_report, render_report_markdown
from app.services.target import apply_target_transform
from app.utils.dataframe import records_to_dataframe


class SimplifiedAutoMindPipeline:
    """A single-agent, stage-based pipeline inspired by AutoMind.

    This class intentionally keeps v1 simple while preserving the boundaries
    that can later become specialized agents: profile, target, preprocess,
    model, audit, and synthesize.
    """

    def run(
        self,
        request: PredictionRequest,
        task_label: str = "prediction",
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        df = records_to_dataframe(request.data or [])
        target_column = request.target_column
        if not target_column:
            raise ValueError("'target_column' is required.")

        dataset_summary, profile_warnings = profile_dataframe(
            df,
            target_column,
            ignore_columns=request.exclude_columns,
        )
        warnings.extend(profile_warnings)

        if df.empty:
            return self._warning_response(task_label, dataset_summary, warnings)

        target, target_name, target_summary, target_warnings = apply_target_transform(
            df,
            target_column,
            request.target_transform,
        )
        warnings.extend(target_warnings)

        if request.task_type == "classification":
            target = target.astype("Int64")

        # Keep only rows with valid transformed targets.
        valid_index = target.dropna().index
        modeling_df = df.loc[valid_index].copy()
        y = target.loc[valid_index]

        preprocessing = prepare_features(
            modeling_df,
            target_column=target_column,
            exclude_columns=request.exclude_columns,
            feature_columns=request.feature_columns,
        )
        warnings.extend(preprocessing.warnings)

        modeling = train_and_evaluate(
            preprocessing.X,
            y,
            preprocessing.preprocessor,
            request.task_type,
            target_name,
            row_metadata=modeling_df,
        )
        warnings.extend(modeling.warnings)

        charts = {
            "feature_importance": modeling.charts.get("feature_importance", []),
            "class_distribution": _class_distribution_for_target(y, target_name)
            if request.task_type == "classification"
            else [],
            "prediction_distribution": modeling.charts.get("prediction_distribution", []),
        }

        insight = generate_insight(task_label, modeling.metrics, modeling.top_features, warnings)
        status = "success" if modeling.metrics else "warning"

        summary = {
            "task": task_label,
            "rows": int(len(modeling_df)),
            "target": target_name,
            "selected_model": modeling.selected_model,
        }
        enriched_target_summary = {
            **target_summary,
            "task_type": request.task_type,
        }
        model_info = {
            "selected_model": modeling.selected_model,
            "candidate_models": modeling.candidate_models,
            "dropped_columns": preprocessing.dropped_columns,
            "numeric_features": preprocessing.numeric_features,
            "categorical_features": preprocessing.categorical_features,
        }
        warnings = sorted(set(warnings))
        report = build_prediction_report(
            title=_report_title(task_label),
            task_label=task_label,
            target_definition=_target_definition(target_name, request.target_transform),
            dataset_summary=dataset_summary,
            target_summary=enriched_target_summary,
            model=model_info,
            metrics=modeling.metrics,
            charts=charts,
            top_features=modeling.top_features,
            sample_predictions=modeling.sample_predictions,
            warnings=warnings,
        )
        llm_requested = is_llm_enabled()
        llm_insights, llm_failure_reason = generate_llm_insights(report)
        if llm_insights:
            report["agent_insights"] = llm_insights
            report["report_markdown"] = render_report_markdown(report)
        elif llm_requested:
            if is_llm_debug_enabled() and llm_failure_reason:
                llm_warning = (
                    f"LLM InsightAgent failed: {llm_failure_reason}; "
                    "rule-based insights were used."
                )
            else:
                llm_warning = "LLM InsightAgent failed; rule-based insights were used."
            warnings = sorted(set(warnings + [llm_warning]))
            report["warnings"] = warnings
            report["report_markdown"] = render_report_markdown(report)

        legacy = {
            "summary": summary,
            "metrics": modeling.metrics,
            "charts": charts,
            "insight": insight,
            "warnings": warnings,
        }
        return {
            "status": status,
            "report": report,
            "legacy": legacy,
            "summary": summary,
            "dataset_summary": dataset_summary,
            "target_summary": enriched_target_summary,
            "model": model_info,
            "metrics": modeling.metrics,
            "charts": charts,
            "top_features": modeling.top_features,
            "insight": insight,
            "warnings": warnings,
        }

    def _warning_response(
        self,
        task_label: str,
        dataset_summary: Dict[str, Any],
        warnings: List[str],
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
        }


def ecommerce_good_review_request(data: Optional[List[Dict[str, Any]]] = None) -> PredictionRequest:
    """Create the default e-commerce good review request."""

    return PredictionRequest(
        data=data,
        target_column="review_score",
        task_type="classification",
        target_transform=TargetTransform(
            type="binary_threshold",
            operator=">=",
            threshold=4,
            positive_name="good_review",
        ),
        exclude_columns=[
            "order_id",
            "customer_id",
            "customer_unique_id",
            "product_id",
            "seller_id",
            "review_id",
            "review_comment_title",
            "review_comment_message",
        ],
        metadata={"source": "wrenai-ecommerce-sample"},
    )


def _class_distribution_for_target(y: pd.Series, target_name: str) -> List[Dict[str, Any]]:
    counts = y.value_counts().sort_index()
    items: List[Dict[str, Any]] = []
    for label, count in counts.items():
        if target_name == "good_review":
            name = "Good review" if int(label) == 1 else "Bad review"
        else:
            name = f"Class {label}"
        items.append({"label": name, "value": int(count)})
    return items


def _report_title(task_label: str) -> str:
    if task_label == "Good review prediction":
        return "AutoMind E-commerce Good Review Prediction Report"
    return f"AutoMind {task_label} Report"


def _target_definition(target_name: str, transform: TargetTransform | None) -> str:
    if target_name == "good_review":
        return "good_review = 1 if review_score >= 4 else 0"
    if transform and transform.type == "binary_threshold":
        return f"{target_name} is derived using {transform.operator} {transform.threshold}"
    return f"Target column: {target_name}"
