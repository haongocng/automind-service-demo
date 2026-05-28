from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException, UploadFile

from app.schemas import PredictionRequest
from app.services.domain_metadata import format_target_label
from app.services.report import render_report_markdown


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TASK_TYPES = {"auto", "classification", "regression"}


async def run_uploaded_prediction(
    *,
    pipeline,
    train_file: UploadFile,
    test_file: Optional[UploadFile],
    target_column: str,
    task_type: str = "auto",
    prediction_goal: Optional[str] = None,
    domain_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the existing AutoMind pipeline against uploaded CSV files."""

    task_type = (task_type or "auto").strip().lower()
    if task_type not in ALLOWED_TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail="unsupported task_type; allowed values are auto, classification, regression",
        )

    target_column = (target_column or "").strip()
    if not target_column:
        raise HTTPException(status_code=400, detail="target_column is required")

    train_df = await _read_upload_csv(train_file, required=True)
    if train_df.empty:
        raise HTTPException(status_code=400, detail="train_file contains an empty dataset")
    if target_column not in train_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"missing target column: {target_column}",
        )

    inferred_task_type = _infer_task_type(train_df[target_column]) if task_type == "auto" else task_type
    if inferred_task_type == "classification":
        unique_count = int(train_df[target_column].dropna().nunique())
        if unique_count < 2:
            raise HTTPException(
                status_code=400,
                detail="target_column must contain at least 2 unique values for classification",
            )
        train_df, label_mapping = _normalize_classification_target(train_df, target_column)
    else:
        label_mapping = None
        train_df[target_column] = pd.to_numeric(train_df[target_column], errors="coerce")
        if train_df[target_column].dropna().empty:
            raise HTTPException(status_code=400, detail="target_column must be numeric for regression")

    test_df = await _read_upload_csv(test_file, required=False) if test_file else None
    if test_df is not None and test_df.empty:
        raise HTTPException(status_code=400, detail="test_file contains an empty dataset")

    display_name = (domain_name or "Custom Prediction").strip() or "Custom Prediction"
    task_label = prediction_goal.strip() if prediction_goal and prediction_goal.strip() else display_name
    metadata = {
        "domain": "custom_prediction",
        "domain_name": display_name,
        "prediction_goal": prediction_goal or "",
        "uploaded_train_filename": train_file.filename or "train.csv",
        "uploaded_test_filename": test_file.filename if test_file else "",
        "label_mapping": label_mapping or {},
    }
    request = PredictionRequest(
        data=_records(train_df),
        target_column=target_column,
        task_type=inferred_task_type,  # type: ignore[arg-type]
        metadata=metadata,
    )

    context = _run_pipeline_with_context(pipeline, request, task_label=task_label)
    response = _build_response(context, task_label)
    _attach_custom_metadata(response, display_name, prediction_goal, inferred_task_type, len(train_df), test_df)

    if test_df is not None:
        sample_predictions = _predict_uploaded_test_rows(context, test_df, target_column)
        response["sample_predictions"] = sample_predictions
        response["test_predictions_included"] = bool(sample_predictions)
        report_prediction_results = response.setdefault("report", {}).setdefault("prediction_results", {})
        report_prediction_results["uploaded_test_predictions"] = sample_predictions[:20]
        response["report"]["report_markdown"] = render_report_markdown(response["report"])
    else:
        response["sample_predictions"] = []
        response["test_predictions_included"] = False

    return response


async def _read_upload_csv(file: Optional[UploadFile], required: bool) -> pd.DataFrame:
    if file is None:
        if required:
            raise HTTPException(status_code=400, detail="train_file is required")
        return pd.DataFrame()

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail=f"invalid CSV file: {filename or 'uploaded file'}")

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="could not read uploaded CSV file") from exc
    finally:
        await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="CSV upload exceeds 10 MB limit")
    if not content.strip():
        raise HTTPException(status_code=400, detail="uploaded CSV file is empty")

    try:
        return pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid CSV content") from exc


def _infer_task_type(target: pd.Series) -> str:
    non_null = target.dropna()
    if non_null.empty:
        raise HTTPException(status_code=400, detail="target_column has no non-null values")
    if pd.api.types.is_bool_dtype(non_null) or pd.api.types.is_object_dtype(non_null):
        return "classification"
    unique_count = int(non_null.nunique())
    if unique_count <= min(20, max(2, int(len(non_null) * 0.1))):
        return "classification"
    return "regression"


def _normalize_classification_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    normalized = df.copy()
    target = normalized[target_column]
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.notna().all():
        normalized[target_column] = numeric.astype(int)
        return normalized, {}

    values = sorted(str(value) for value in target.dropna().unique())
    mapping = {value: index for index, value in enumerate(values)}
    normalized[target_column] = target.map(lambda value: mapping.get(str(value)) if pd.notna(value) else None)
    return normalized, mapping


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _run_pipeline_with_context(
    pipeline,
    request: PredictionRequest,
    task_label: str,
) -> Dict[str, Any]:
    orchestrator = pipeline.orchestrator
    context: Dict[str, Any] = {
        "request": request,
        "task_label": task_label,
        "warnings": [],
        "agent_trace": [],
    }
    orchestrator.data_profiler.run(context)
    orchestrator.preprocessing.run(context)
    orchestrator.modeling.run(context)
    orchestrator.model_audit.run(context)
    orchestrator.report.run(context)
    orchestrator.insight.run(context)
    return context


def _build_response(context: Dict[str, Any], task_label: str) -> Dict[str, Any]:
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
    context["report"]["report_markdown"] = context["report"].get("report_markdown", "")
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


def _attach_custom_metadata(
    response: Dict[str, Any],
    domain_name: str,
    prediction_goal: Optional[str],
    task_type: str,
    train_rows: int,
    test_df: Optional[pd.DataFrame],
) -> None:
    test_rows = int(len(test_df)) if test_df is not None else 0
    response["domain"] = "custom_prediction"
    response["domain_name"] = domain_name
    response["prediction_goal"] = prediction_goal or ""
    response["task_type"] = task_type
    response["train_rows"] = train_rows
    response["test_rows"] = test_rows
    report = response.get("report", {})
    report.setdefault("custom_prediction", {})
    report["custom_prediction"].update(
        {
            "domain": "custom_prediction",
            "domain_name": domain_name,
            "prediction_goal": prediction_goal or "",
            "task_type": task_type,
            "train_rows": train_rows,
            "test_rows": test_rows,
        }
    )
    target_metadata = report.setdefault("target_metadata", {})
    target_metadata["domain_name"] = domain_name
    target_metadata["prediction_goal"] = prediction_goal or ""
    report["report_markdown"] = render_report_markdown(report)


def _predict_uploaded_test_rows(
    context: Dict[str, Any],
    test_df: pd.DataFrame,
    target_column: str,
) -> List[Dict[str, Any]]:
    modeling = context["modeling"]
    fitted_pipeline = getattr(modeling, "fitted_pipeline", None)
    if fitted_pipeline is None:
        context.setdefault("warnings", []).append("Test predictions were skipped because no fitted model is available.")
        return []

    train_features = list(context["preprocessing"].X.columns)
    missing = [column for column in train_features if column not in test_df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail="train/test feature mismatch; test_file is missing columns: " + ", ".join(missing),
        )

    X_test = test_df[train_features].copy()
    try:
        predictions = fitted_pipeline.predict(X_test)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="could not generate predictions for test_file") from exc

    probabilities = None
    if context["request"].task_type == "classification" and hasattr(fitted_pipeline, "predict_proba"):
        try:
            proba = fitted_pipeline.predict_proba(X_test)
            if proba is not None and len(proba.shape) == 2 and proba.shape[1] >= 2:
                probabilities = proba[:, 1]
        except Exception:
            probabilities = None

    samples: List[Dict[str, Any]] = []
    for position, (index, prediction) in enumerate(zip(test_df.index, predictions)):
        item: Dict[str, Any] = {
            "row_index": int(index) if isinstance(index, int) else str(index),
            "predicted": _format_prediction(prediction, target_column, context["request"].task_type),
        }
        if probabilities is not None:
            item[f"probability_{target_column}"] = round(float(probabilities[position]), 4)
        samples.append(item)
        if len(samples) >= 20:
            break
    return samples


def _format_prediction(value: Any, target_column: str, task_type: str) -> Any:
    if task_type == "classification":
        return format_target_label(value, target_column, predicted=True)
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return value
