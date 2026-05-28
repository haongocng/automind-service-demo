from __future__ import annotations

from typing import Any, Dict

from app.agents.base import BaseAgent
from app.services.preprocessing import prepare_features
from app.services.target import apply_target_transform


class PreprocessingAgent(BaseAgent):
    name = "PreprocessingAgent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request = context["request"]
        df = context["df"]
        target, target_name, target_summary, target_warnings = apply_target_transform(
            df,
            request.target_column,
            request.target_transform,
        )
        context.setdefault("warnings", []).extend(target_warnings)

        if request.task_type == "classification":
            target = target.astype("Int64")

        valid_index = target.dropna().index
        modeling_df = df.loc[valid_index].copy()
        y = target.loc[valid_index]

        preprocessing = prepare_features(
            modeling_df,
            target_column=request.target_column,
            exclude_columns=request.exclude_columns,
            feature_columns=request.feature_columns,
        )
        context.setdefault("warnings", []).extend(preprocessing.warnings)
        context.update(
            {
                "target": target,
                "target_name": target_name,
                "target_summary": target_summary,
                "modeling_df": modeling_df,
                "y": y,
                "preprocessing": preprocessing,
            }
        )
        self.add_trace(context, "success", "Prepared target and features.")
        return context
