from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agents import AutoMindOrchestrator
from app.schemas import PredictionRequest, TargetTransform


class SimplifiedAutoMindPipeline:
    """Public pipeline facade kept for backward compatibility.

    The implementation now delegates to a lightweight multi-agent orchestrator
    while preserving the existing response contract used by FastAPI endpoints
    and the WrenAI frontend.
    """

    def __init__(self) -> None:
        self.orchestrator = AutoMindOrchestrator()

    def run(
        self,
        request: PredictionRequest,
        task_label: str = "prediction",
    ) -> Dict[str, Any]:
        return self.orchestrator.run(request, task_label=task_label)


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


def heart_disease_request(data: Optional[List[Dict[str, Any]]] = None) -> PredictionRequest:
    """Create the default Heart Disease classification request."""

    return PredictionRequest(
        data=data,
        target_column="HeartDisease",
        task_type="classification",
        exclude_columns=[],
        metadata={"source": "prepared-heart-disease-csv", "domain": "heart_disease"},
    )

