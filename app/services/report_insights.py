"""Data-grounded explanations for saved analysis reports."""
from __future__ import annotations

import numpy as np
import pandas as pd


def number(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def insight(ident, title, detail, *evidence):
    return {"id": ident, "title": title, "detail": detail, "evidenceIds": list(evidence)}


def data_limits(frame, features, target=None):
    limits = []
    missing = [(c, int(frame[c].isna().sum())) for c in features if frame[c].isna().any()]
    if missing:
        limits.append("Incomplete inputs: " + "; ".join(f"{c} is missing in {n:,} rows ({n/len(frame):.1%})" for c, n in missing[:3]) + ". Comparisons involving these fields depend on how missing responses are handled.")
    for c in features:
        if c.lower() in ("age", "cholesterol", "restingbp") and pd.api.types.is_numeric_dtype(frame[c]):
            zeros = int(frame[c].eq(0).sum())
            if zeros:
                limits.append(f"{c} contains {zeros:,} zero-valued records ({zeros/len(frame):.1%}). Verify whether zero is a genuine measurement or an unavailable value in the source before relying on this field.")
    if sum(c.startswith("I ") for c in frame.columns) >= 3:
        limits.append("These are self-reported survey answers. Shared response styles and questions collected together can create strong associations without demonstrating improvement in learning or a causal effect of technology.")
        limits.append("The survey has no verified respondent or school identifier. Similar answer patterns may come from different people, and the sample cannot establish representativeness of all students or schools.")
    if target:
        values = pd.to_numeric(frame[target], errors="coerce")
        unique = values.dropna().unique()
        if values.notna().all() and 3 <= len(unique) <= 10 and np.all(unique == np.floor(unique)):
            limits.append(f"{target} takes {len(unique)} discrete values from {number(values.min())} to {number(values.max())}. Regression treats the gaps as equal; a fractional prediction is an estimated score, not an additional response category.")
    return limits


def supervised_insights(task, target, ytrain, ytest, prediction, features, frame, training):
    findings, insights = [], []
    limits = data_limits(frame, features, target if task == "regression" else None)
    if task == "classification":
        accuracy = float(np.mean(np.asarray(ytest) == prediction))
        majority = str(ytrain.value_counts().idxmax())
        baseline = float(np.mean(ytest == majority))
        correct = int(np.sum(np.asarray(ytest) == prediction))
        detail = f"{correct:,} of {len(ytest):,} evaluation records were classified correctly ({accuracy:.1%}); {len(ytest)-correct:,} were misclassified. Always choosing the most frequent training class ('{majority}') would score {baseline:.1%} on these same records. "
        detail += (f"The model improves accuracy by {(accuracy-baseline)*100:.1f} percentage points." if accuracy > baseline else "The model does not improve on this simple accuracy benchmark; review class-level errors before using it.")
        insights.append(insight("accuracy", "Performance against a simple benchmark", detail, "confusion"))
        class_details = []
        recall_rows = []
        for label in sorted(set(ytest) | set(prediction)):
            actual = np.asarray(ytest) == label
            predicted = prediction == label
            tp = int(np.sum(actual & predicted))
            support = int(actual.sum())
            missed = support - tp
            fp = int(np.sum(~actual & predicted))
            recall = tp / support if support else 0
            precision = tp / int(predicted.sum()) if predicted.sum() else 0
            class_details.append(f"Class '{label}': {tp:,} of {support:,} actual records detected ({recall:.1%} recall), {missed:,} missed and {fp:,} other records incorrectly assigned to it ({precision:.1%} precision).")
            recall_rows.append((recall, support, label, missed))
        weakest = min((r for r in recall_rows if r[1]), default=(0, 0, "", 0))
        insights.append(insight("class-errors", f"Review errors for class '{weakest[2]}'", " ".join(class_details) + " The confusion matrix shows which recorded outcomes are most often confused; the consequence of an error depends on the task.", "confusion"))
        counts = ytrain.value_counts()
        insights.append(insight("class-balance", "Class balance affects what the headline score means", "; ".join(f"Training class '{c}': {n:,} records ({n/len(ytrain):.1%})" for c, n in counts.items()) + ". Macro F1 gives every class equal weight, so compare it with accuracy when assessing uneven class performance.", "confusion", "model-selection"))
        interpretation = detail + " " + " ".join(class_details)
        limits.insert(0, f"Quality was measured on {len(ytest):,} held-out records from this selected population. Small per-class counts can make recall and precision unstable; performance on a different population remains unmeasured.")
        limits.append("The target describes the recorded outcome. The report does not establish a decision threshold, calibrated probability, or the cost of false positive and false negative predictions.")
    else:
        actual = np.asarray(ytest, dtype=float)
        residual = actual - prediction
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        baseline = float(np.sqrt(np.mean((actual - float(ytrain.mean()))**2)))
        bias = float(np.mean(prediction - actual))
        relative = (baseline-rmse)/baseline if baseline else 0
        detail = f"The typical absolute error is {number(mae)} units of '{target}'; larger errors raise RMSE to {number(rmse)}. Predicting the training mean for every record gives RMSE {number(baseline)} on the same evaluation set. "
        detail += (f"The model reduces that error by {relative:.1%}." if rmse < baseline else "The model does not improve on that benchmark; these predictions need further review.")
        insights.append(insight("error-size", "How accurate are the predictions?", detail, "actual-predicted"))
        insights.append(insight("bias", "Check both typical and unusually large errors", f"Mean predicted-minus-observed difference is {number(bias)}: {'overestimation' if bias > 0 else 'underestimation' if bias < 0 else 'no average directional bias'} on this set. Half of absolute errors are at most {number(np.median(np.abs(residual)))}; 90% are at most {number(np.quantile(np.abs(residual), .9))}. A small average bias can still conceal large individual errors.", "residuals"))
        discrete = 3 <= len(np.unique(actual)) <= 10 and np.all(actual == np.floor(actual))
        distribution = [{"label": number(v), "value": int(np.sum(actual == v))} for v in sorted(np.unique(actual))] if discrete else []
        if distribution:
            population = pd.to_numeric(frame[target])
            levels = sorted(population.unique())
            population_detail = f"Across all {len(population):,} selected records, the mean score is {number(population.mean())} and the median is {number(population.median())}. " + "; ".join(f"Score {number(v)}: {int(population.eq(v).sum()):,} records ({float(population.eq(v).mean()):.1%})" for v in levels) + ". These counts describe the observed ratings in this sample; a survey rating does not demonstrate subsequent behavior."
            insights.insert(0, insight("reported-scores", "The target distribution in this sample", population_detail, "population-target"))
            findings.append({"id": "population-target", "title": "Read the reported scores before interpreting predictions", "interpretation": population_detail, "evidenceIds": ["population-target"]})
            insights.append(insight("score-coverage", "Coverage of the observed response levels", f"Observed evaluation scores span {number(actual.min())}–{number(actual.max())}; {float(np.mean(np.abs(residual) <= 1)):.1%} of predictions are within one score point. " + "; ".join(f"Score {v['label']}: {v['value']} evaluation responses" for v in distribution) + ". Rare response levels have less evidence for judging prediction quality.", "target-distribution"))
        else:
            insights.append(insight("coverage", "Interpret error in the target's own scale", f"Evaluation values range from {number(actual.min())} to {number(actual.max())}, with a median of {number(np.median(actual))}. Compare the absolute error with a useful tolerance for this measurement rather than reading R² as a percentage of correct predictions.", "actual-predicted"))
        interpretation = detail + " Inspect how predictions track low and high observed values; narrow-looking predictions can miss unusual outcomes even when average error is acceptable."
        associations = []
        for c in features:
            if pd.api.types.is_numeric_dtype(training[c]):
                values = pd.to_numeric(training[c], errors="coerce")
                if values.nunique() > 1:
                    correlation = float(values.corr(ytrain))
                    if np.isfinite(correlation):
                        associations.append((abs(correlation), c, correlation))
        if associations:
            association_detail = "Among training records, the strongest observed numerical associations with the target are " + "; ".join(f"'{c}' ({correlation:+.2f} correlation)" for _, c, correlation in sorted(associations, reverse=True)[:3]) + ". These associations help explain which answers vary together; they are not causal effects or a measure of the fitted model's feature importance."
            insights.append(insight("associated-inputs", "Which recorded characteristics vary with the target?", association_detail))
        findings.append({"id": "residual-pattern", "title": "Where predictions overestimate or underestimate", "interpretation": next(item['detail'] for item in insights if item['id'] == 'bias') + " Values above zero in this residual chart mean observed scores were higher than predicted; values below zero mean predictions were too high. Check whether spread changes with the predicted score before setting an acceptable error tolerance.", "evidenceIds": ["residuals"]})
        limits.insert(0, f"Evaluation covers {len(ytest):,} records from the selected dataset, not a future cohort. The observed target range is {number(actual.min())}–{number(actual.max())}; accuracy outside that range or under a changed response distribution has not been measured.")
    limits.append("Predictors must be available when the prediction would be made. Fields collected after the outcome, or survey answers expressing nearly the same concept as the target, can make measured performance overly optimistic.")
    return insights, interpretation, findings, limits
