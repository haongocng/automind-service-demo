from __future__ import annotations

from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse, PredictionRequest
from app.services.automind_agent import (
    SimplifiedAutoMindPipeline,
    ecommerce_good_review_request,
)
from app.services.demo_data import ecommerce_good_review_records

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
