from __future__ import annotations

from typing import Any, Dict

from app.agents.base import BaseAgent
from app.services.profiler import profile_dataframe
from app.utils.dataframe import records_to_dataframe


class DataProfilerAgent(BaseAgent):
    name = "DataProfilerAgent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request = context["request"]
        target_column = request.target_column
        if not target_column:
            raise ValueError("'target_column' is required.")

        df = records_to_dataframe(request.data or [])
        dataset_summary, profile_warnings = profile_dataframe(
            df,
            target_column,
            ignore_columns=request.exclude_columns,
        )
        context["df"] = df
        context["dataset_summary"] = dataset_summary
        context.setdefault("warnings", []).extend(profile_warnings)
        self.add_trace(context, "success", "Profiled dataset.")
        return context
