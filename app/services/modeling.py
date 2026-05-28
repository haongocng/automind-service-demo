from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.services.domain_metadata import format_target_label
from app.services.predictions import build_sample_predictions


@dataclass
class ModelingResult:
    selected_model: Optional[str]
    candidate_models: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    charts: Dict[str, List[Dict[str, Any]]]
    top_features: List[Dict[str, Any]]
    sample_predictions: List[Dict[str, Any]]
    warnings: List[str]


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    task_type: str,
    target_name: str,
    row_metadata: Optional[pd.DataFrame] = None,
) -> ModelingResult:
    """Train candidate models and return metrics plus chart-ready data."""

    warnings: List[str] = []
    if X.empty or X.shape[1] == 0:
        return _empty_result(["No usable features are available for modeling."])
    if len(y) < 20:
        return _empty_result(["Dataset has fewer than 20 labeled rows; skipping model training."])

    y = y.astype(int) if task_type == "classification" else pd.to_numeric(y, errors="coerce")
    valid_index = y.dropna().index
    X = X.loc[valid_index]
    y = y.loc[valid_index]

    if task_type == "classification":
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return _empty_result(["Target has only one class; skipping model training."])
        if class_counts.min() < 2:
            warnings.append("At least one class has fewer than 2 rows; stratified split is disabled.")
        if class_counts.min() / len(y) < 0.15:
            warnings.append("Target is imbalanced; F1-score is more informative than accuracy.")

    test_size = 0.25 if len(y) >= 40 else 0.3
    stratify = None
    if task_type == "classification" and y.value_counts().min() >= 2:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    if len(y_test) < 10:
        warnings.append("Test set has fewer than 10 rows; metrics may be unstable.")

    candidates = _classification_candidates() if task_type == "classification" else _regression_candidates()
    candidate_results: List[Dict[str, Any]] = []
    best_pipeline: Optional[Pipeline] = None
    best_name: Optional[str] = None
    best_score = -np.inf
    best_predictions = None
    best_probabilities = None

    for name, estimator in candidates:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = _positive_class_probabilities(pipeline, X_test, task_type)
            score = _selection_score(y_test, predictions, task_type)
            candidate_results.append({"name": name, "score": round(float(score), 4)})
            if score > best_score:
                best_score = score
                best_name = name
                best_pipeline = pipeline
                best_predictions = predictions
                best_probabilities = probabilities
        except Exception as exc:  # Keep one failed model from breaking the demo.
            candidate_results.append({"name": name, "score": None, "error": str(exc)})
            warnings.append(f"Model '{name}' failed and was skipped: {exc}")

    if best_pipeline is None or best_predictions is None:
        return _empty_result(warnings + ["All candidate models failed."])

    metrics, charts = _audit_predictions(y_test, best_predictions, task_type, target_name)
    top_features = _feature_importance(best_pipeline)
    sample_predictions = build_sample_predictions(
        row_metadata,
        y_test,
        best_predictions,
        best_probabilities,
        target_name,
    )
    if metrics.get("accuracy", 0) >= 0.98 and task_type == "classification":
        warnings.append("Performance is suspiciously high; check for leakage columns.")

    charts["feature_importance"] = [
        {"feature": item["feature"], "value": item["importance"]} for item in top_features
    ]

    return ModelingResult(
        selected_model=best_name,
        candidate_models=candidate_results,
        metrics=metrics,
        charts=charts,
        top_features=top_features,
        sample_predictions=sample_predictions,
        warnings=warnings,
    )


def _classification_candidates():
    return [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            ),
        ),
    ]


def _regression_candidates():
    return [
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2),
        )
    ]


def _selection_score(y_true, predictions, task_type: str) -> float:
    if task_type == "classification":
        return f1_score(y_true, predictions, zero_division=0)
    rmse = mean_squared_error(y_true, predictions, squared=False)
    return -rmse


def _audit_predictions(y_true, predictions, task_type: str, target_name: str):
    if task_type == "classification":
        labels = sorted(pd.Series(y_true).dropna().unique().tolist())
        cm = confusion_matrix(y_true, predictions, labels=labels).tolist()
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
            "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
            "confusion_matrix": cm,
        }
        charts = {
            "class_distribution": _distribution_chart(y_true, target_name, observed=True),
            "prediction_distribution": _distribution_chart(predictions, target_name, observed=False),
        }
        return metrics, charts

    rmse = mean_squared_error(y_true, predictions, squared=False)
    metrics = {
        "mae": round(float(mean_absolute_error(y_true, predictions)), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2_score(y_true, predictions)), 4),
    }
    return metrics, {}


def _positive_class_probabilities(pipeline: Pipeline, X_test: pd.DataFrame, task_type: str):
    if task_type != "classification" or not hasattr(pipeline, "predict_proba"):
        return None
    try:
        probabilities = pipeline.predict_proba(X_test)
    except Exception:
        return None
    if probabilities is None or len(probabilities.shape) != 2 or probabilities.shape[1] < 2:
        return None
    return probabilities[:, 1]


def _distribution_chart(values, target_name: str, observed: bool) -> List[Dict[str, Any]]:
    counts = pd.Series(values).value_counts().sort_index()
    items: List[Dict[str, Any]] = []
    for label, count in counts.items():
        items.append({"label": _class_label(label, target_name, observed), "value": int(count)})
    return items


def _class_label(label: Any, target_name: str, observed: bool) -> str:
    return format_target_label(label, target_name, predicted=not observed)


def _feature_importance(pipeline: Pipeline, top_n: int = 10) -> List[Dict[str, Any]]:
    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline)

    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or len(values) != len(names):
        return []

    order = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": names[index], "importance": round(float(values[index]), 4)}
        for index in order
        if float(values[index]) > 0
    ]


def _feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        raw_names = preprocessor.get_feature_names_out()
    except Exception:
        return []
    names: List[str] = []
    for name in raw_names:
        clean = str(name)
        for prefix in ("numeric__", "categorical__"):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        names.append(clean)
    return names


def _empty_result(warnings: List[str]) -> ModelingResult:
    return ModelingResult(
        selected_model=None,
        candidate_models=[],
        metrics={},
        charts={"feature_importance": [], "class_distribution": [], "prediction_distribution": []},
        top_features=[],
        sample_predictions=[],
        warnings=warnings,
    )
