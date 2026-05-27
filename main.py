from __future__ import annotations

from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse, PredictionRequest
from app.services.automind_agent import (
    SimplifiedAutoMindPipeline,
    ecommerce_good_review_request,
    heart_disease_request,
)
from app.services.demo_data import (
    ecommerce_good_review_records,
    get_heart_disease_train_records,
    get_heart_disease_test_records,
)
from app.services.report import render_report_markdown

VERSION = "0.1.0"

app = FastAPI(
    title="AutoMind-service",
    version=VERSION,
    description="Lightweight AutoMind-style prediction service for WrenAI demos.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = SimplifiedAutoMindPipeline()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AutoMind-service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "demo_prediction": "/predict/ecommerce-good-review",
        "heart_disease_prediction": "/predict/heart-disease",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="AutoMind-service", version=VERSION)


@app.post("/predict")
def predict(request: PredictionRequest):
    """Generic reusable endpoint for future domains such as HeartDisease."""

    try:
        return pipeline.run(request, task_label=request.metadata.get("task", "Prediction"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/ecommerce-good-review")
def predict_ecommerce_good_review(
    request: Optional[PredictionRequest] = Body(default=None),
):
    """Demo endpoint for predicting whether an order receives a good review."""

    try:
        # Swagger UI may submit its generated placeholder body (for example
        # {"additionalProp1": {}}). For this demo endpoint, only treat the
        # request body as real e-commerce data when it contains review_score.
        incoming_records = request.data if request and request.data else None
        has_ecommerce_target = bool(
            incoming_records and any("review_score" in row for row in incoming_records)
        )
        records = incoming_records if has_ecommerce_target else ecommerce_good_review_records()
        default_request = ecommerce_good_review_request(records)

        if request:
            default_request.exclude_columns = list(
                set(default_request.exclude_columns + request.exclude_columns)
            )
            if request.feature_columns and has_ecommerce_target:
                default_request.feature_columns = request.feature_columns
            default_request.metadata.update(request.metadata)

        result = pipeline.run(default_request, task_label="Good review prediction")
        if incoming_records and not has_ecommerce_target:
            result["warnings"].append(
                "Request body did not contain review_score; built-in e-commerce demo data was used."
            )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


HEART_DISEASE_LIMITATIONS = [
    "This Heart Disease workflow is for demonstration and research only.",
    "The output is not medical advice, diagnosis, treatment guidance, or a clinical decision system.",
    "Real clinical deployment would require expert validation, calibration, bias assessment, and regulatory review.",
]


@app.post("/predict/heart-disease")
def predict_heart_disease(
    request: Optional[PredictionRequest] = Body(default=None),
):
    """Prepared Heart Disease binary classification demo endpoint."""

    try:
        incoming_records = request.data if request and request.data else None
        if incoming_records and not any("HeartDisease" in row for row in incoming_records):
            raise ValueError(
                "Provided Heart Disease data must include the 'HeartDisease' target column. "
                "Prediction-only upload for unlabeled rows is not implemented yet."
            )

        records = incoming_records if incoming_records else get_heart_disease_train_records()
        default_request = heart_disease_request(records)

        if request:
            default_request.exclude_columns = list(
                set(default_request.exclude_columns + request.exclude_columns)
            )
            if request.feature_columns and incoming_records:
                default_request.feature_columns = request.feature_columns
            default_request.metadata.update(request.metadata)

        result = pipeline.run(default_request, task_label="Heart disease classification")
        _attach_heart_disease_disclaimer(result)
        result["heart_disease_dataset"] = {
            "train_rows": len(records),
            "test_rows": len(get_heart_disease_test_records()) if not incoming_records else 0,
            "test_predictions_included": False,
            "note": (
                "The prepared unlabeled heart_test.csv is loaded for availability checks, "
                "but prediction-only test output is deferred until the pipeline exposes a fitted model."
            ),
        }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _attach_heart_disease_disclaimer(result):
    report = result.get("report") if isinstance(result, dict) else None
    if not report:
        return

    limitations = list(report.get("limitations", []))
    for item in HEART_DISEASE_LIMITATIONS:
        if item not in limitations:
            limitations.append(item)
    report["limitations"] = limitations

    recommendations = list(report.get("recommendations", []))
    clinical_recommendation = (
        "Do not use this demo output for patient care; involve qualified clinical experts for any medical workflow."
    )
    if clinical_recommendation not in recommendations:
        recommendations.append(clinical_recommendation)
    report["recommendations"] = recommendations

    report["report_markdown"] = render_report_markdown(report)
