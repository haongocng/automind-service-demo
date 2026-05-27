from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TaskType = Literal["classification", "regression"]
TransformType = Literal["none", "binary_threshold"]


class TargetTransform(BaseModel):
    """Optional target transformation configuration."""

    type: TransformType = "none"
    operator: Optional[Literal[">=", ">", "<=", "<", "=="]] = None
    threshold: Optional[float] = None
    positive_name: Optional[str] = None


class PredictionRequest(BaseModel):
    """Generic prediction request used by all demo domains."""

    data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="JSON records. If omitted for a demo endpoint, built-in demo data may be used.",
    )
    target_column: Optional[str] = None
    task_type: TaskType = "classification"
    target_transform: Optional[TargetTransform] = None
    exclude_columns: List[str] = Field(default_factory=list)
    feature_columns: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
