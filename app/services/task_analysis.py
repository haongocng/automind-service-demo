"""Real analysis tasks for the Explorer bridge; no LLM or demo-data fallback.

Model selection sees training folds only. The final holdout (or separate
labelled test dataset) is evaluated once after the candidate has been selected.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, TimeSeriesSplit, KFold, train_test_split
from sklearn.pipeline import Pipeline

from app.services.modeling import _classification_candidates, _regression_candidates
from app.services.preprocessing import prepare_features


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["clustering", "classification", "regression"]
    goal: str = Field(min_length=1, max_length=2000)
    data: list[dict[str, Any]]
    testData: list[dict[str, Any]] | None = None
    fields: list[dict[str, Any]]
    target: str | None = None
    features: list[str] | None = None
    evaluation: Literal["automatic", "separate"] = "automatic"
    split: Literal["auto", "random", "group", "time"] = "auto"
    groupColumn: str | None = None
    dateColumn: str | None = None
    clusters: int | None = Field(default=None, ge=2, le=12)
    rowGrain: str = "One source row"


class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Reuse our existing feature rules, learned independently in every fold."""
    def __init__(self, target="", excludes=(), features=None):
        self.target = target
        self.excludes = excludes
        self.features = features

    def fit(self, X, y=None):
        prepared = prepare_features(X, self.target, list(self.excludes), self.features)
        self.columns_ = list(prepared.X.columns)
        # All-null columns are unsuitable; learn that decision from this fold.
        self.columns_ = [c for c in self.columns_ if X[c].notna().any()]
        if not self.columns_:
            raise ValueError("No usable features remain in the training partition. Select a different view or features.")
        prepared = prepare_features(X[self.columns_], self.target, list(self.excludes), self.columns_)
        self.preprocessor_ = prepared.preprocessor.fit(prepared.X)
        self.warnings_ = prepared.warnings
        return self

    def transform(self, X):
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError("Evaluation data is missing feature columns: " + ", ".join(sorted(missing)))
        return self.preprocessor_.transform(X[self.columns_])

    def get_feature_names_out(self, input_features=None):
        return self.preprocessor_.get_feature_names_out()


def visual(vid, title, values, mark, encoding, caption):
    return {
        "id": vid, "title": title, "kind": "chartSpec", "caption": caption,
        "altText": f"{title}. {caption}",
        "chartSpec": {"mark": mark, "encoding": encoding, "data": {"values": values}},
    }


def bar(vid, title, values, caption):
    return visual(vid, title, values, {"type": "bar", "color": "#477dff"}, {
        "x": {"field": "value", "type": "quantitative", "title": "Value"},
        "y": {"field": "label", "type": "nominal", "sort": "-x", "title": None},
        "tooltip": [{"field": "label"}, {"field": "value", "type": "quantitative"}],
    }, caption)


def _frame(records):
    if not records or len(records) > 20000:
        raise ValueError("This ML analysis supports 5–20,000 rows in the selected scope. Narrow the scope or select another view; no rows are silently sampled.")
    df = pd.DataFrame(records).replace({None: np.nan})
    if len(df) < 5:
        raise ValueError("At least five rows are required for this analysis.")
    return df


def _labels(df, target, task):
    if not target or target not in df:
        raise ValueError("Choose an available target column.")
    if df[target].isna().any():
        raise ValueError("Target contains missing labels. Filter to labelled rows before running; labels are not generated or imputed.")
    if task == "regression":
        y = pd.to_numeric(df[target], errors="coerce")
        if y.isna().any() or not np.isfinite(y).all() or y.nunique() < 2:
            raise ValueError("Regression requires a finite, varying numeric target.")
        return y
    y = df[target].astype(str)
    if y.nunique() < 2 or y.nunique() > 50:
        raise ValueError("Classification requires 2–50 observed target classes; select a category rather than an identifier or continuous value.")
    return y


def _split(request, df, y):
    """Resolve evaluation from real rows. Repeated entity IDs stay together."""
    strategy = request.split
    ids = [f["name"] for f in request.fields if f.get("role") == "identifier" and f["name"] in df]
    repeated = [c for c in ids if 2 <= df[c].nunique() < len(df) and not df[c].isna().any()]
    if strategy == "auto":
        future_goal = any(word in request.goal.lower() for word in ("future", "next period", "next month", "forecast"))
        if future_goal and not request.dateColumn:
            raise ValueError("A future-period prediction goal requires a dated view and time-based evaluation. Choose a compatible view or revise the goal.")
        strategy = "time" if future_goal else "group" if repeated else "random"
    groups = None
    group_column = request.groupColumn
    if strategy == "group":
        group_column = group_column or (repeated[0] if repeated else None)
        if not group_column or group_column not in df or df[group_column].isna().any():
            raise ValueError("Choose a non-null entity/group column for group-aware evaluation.")
        groups = df[group_column].astype(str)
        if groups.nunique() < 5:
            raise ValueError("Group-aware evaluation requires at least five independent groups.")
    if strategy == "time":
        if not request.dateColumn or request.dateColumn not in df:
            raise ValueError("Time-based evaluation requires a date column in this view.")
        dates = pd.to_datetime(df[request.dateColumn], errors="coerce", utc=True)
        if dates.isna().any() or dates.nunique() < 5:
            raise ValueError("Time-based evaluation requires at least five valid time periods and no missing dates.")
        order = dates.sort_values(kind="stable").index.to_numpy()
        # Do not divide identical timestamps between train and final holdout.
        boundary = dates.loc[order[int(len(order) * .75)]]
        train = order[(dates.loc[order] < boundary).to_numpy()]
        test = order[(dates.loc[order] >= boundary).to_numpy()]
        # Entity leakage takes precedence over convenience.
        for c in repeated:
            if set(df.loc[train, c]) & set(df.loc[test, c]):
                raise ValueError("Time split shares entities between training and test. Select a group-aware split or an independent entity-level view.")
    elif strategy == "group":
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(df, y, groups))
    else:
        if request.split == "random" and repeated:
            raise ValueError("Repeated entity IDs require group-aware evaluation; use Automatic strategy or Group-aware.")
        stratify = y if request.task == "classification" else None
        train, test = train_test_split(np.arange(len(df)), test_size=.25, random_state=42, stratify=stratify)
    return np.array(train), np.array(test), strategy, groups, group_column


def _cv(task, strategy, X, y, groups, date_column=None):
    if strategy == "group":
        return list(GroupKFold(n_splits=min(3, groups.nunique())).split(X, y, groups))
    if strategy == "time":
        dates = pd.to_datetime(X[date_column], utc=True)
        unique = np.array(sorted(dates.unique()))
        splits = []
        for a, b in TimeSeriesSplit(n_splits=3).split(unique):
            splits.append((np.flatnonzero(dates.isin(unique[a])), np.flatnonzero(dates.isin(unique[b]))))
        return splits
    if task == "classification":
        folds = min(3, int(y.value_counts().min()))
        if folds < 2:
            raise ValueError("Training needs at least two rows of every class for independent cross-validation. Add labelled data or change the scope.")
        return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(X, y))
    return list(KFold(n_splits=3, shuffle=True, random_state=42).split(X))


def analyze_task(request: TaskRequest):
    df = _frame(request.data)
    fields = {f["name"]: f for f in request.fields}
    if set(df.columns) != set(fields):
        raise ValueError("Snapshot fields do not match the verified schema.")
    excludes = tuple(f["name"] for f in request.fields if f.get("role") in ("identifier", "date", "other"))
    if request.features is not None:
        if not request.features or set(request.features) - set(df.columns) or set(request.features) & (set(excludes) | {request.target}):
            raise ValueError("Select usable feature columns; identifiers, dates and the target cannot be used as features.")
    if request.task == "clustering":
        return _clustering(request, df, excludes)
    if len(df) < 24:
        raise ValueError("Supervised evaluation requires at least 24 labelled rows for training, model selection and a final holdout.")
    if not request.target or request.target in excludes:
        raise ValueError("Target must be a meaningful category or numeric field, rather than an identifier/date.")
    y = _labels(df, request.target, request.task)
    if request.task == "classification" and y.value_counts().min() < 4:
        raise ValueError("At least four labelled rows per class are required for training, selection and final evaluation.")
    if request.evaluation == "separate":
        test = _frame(request.testData or [])
        yt = _labels(test, request.target, request.task)
        if set(df.columns) - set(test.columns):
            raise ValueError("Separate evaluation data is missing training columns. Use a compatible labelled dataset.")
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) != pd.api.types.is_numeric_dtype(test[c]):
                raise ValueError(f"Evaluation column '{c}' has an incompatible type.")
        left = set(pd.util.hash_pandas_object(df.sort_index(axis=1), index=False).tolist())
        right = set(pd.util.hash_pandas_object(test[df.columns].sort_index(axis=1), index=False).tolist())
        if left & right:
            raise ValueError("Training and evaluation contain duplicate rows. Choose an independent test dataset.")
        repeated_ids = [c for c in excludes if fields[c].get("role") == "identifier"]
        for c in repeated_ids:
            if set(df[c].dropna().astype(str)) & set(test[c].dropna().astype(str)):
                raise ValueError(f"Training and test share entity IDs in '{c}'. Use independent entities.")
        if request.task == "classification" and set(yt) - set(y):
            raise ValueError("Test labels contain classes absent from training.")
        train_idx, _, strategy, groups, group_column = _split(request, df, y)
        # Model selection uses all supplied training rows; only the independent test is final.
        train_idx = np.arange(len(df))
        if strategy == "time":
            train_idx = pd.to_datetime(df[request.dateColumn], utc=True).sort_values().index.to_numpy()
            if request.dateColumn not in test or pd.to_datetime(test[request.dateColumn], errors="coerce", utc=True).isna().any():
                raise ValueError("Time-based evaluation requires valid dates in the separate test dataset.")
            if pd.to_datetime(test[request.dateColumn], utc=True).min() <= pd.to_datetime(df[request.dateColumn], utc=True).max():
                raise ValueError("The separate test must follow all training periods for time-based evaluation.")
        Xtrain, ytrain, Xtest, ytest = df.iloc[train_idx], y.iloc[train_idx], test, yt
        groups_train = groups.iloc[train_idx] if groups is not None else None
        evaluation_label = "Separate labelled dataset"
    else:
        if request.testData is not None:
            raise ValueError("Automatic split does not accept a separate test dataset.")
        train_idx, test_idx, strategy, groups, group_column = _split(request, df, y)
        Xtrain, ytrain, Xtest, ytest = df.iloc[train_idx], y.iloc[train_idx], df.iloc[test_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx] if groups is not None else None
        evaluation_label = "Independent 25% holdout" if strategy != "time" else "Final time periods held out"
    if len(Xtrain) < 12 or len(Xtest) < 5 or (request.task == "classification" and set(ytest) - set(ytrain)):
        raise ValueError("The resolved split has insufficient training/test rows or unseen holdout classes. Change the scope or evaluation strategy.")
    folds = _cv(request.task, strategy, Xtrain, ytrain, groups_train, request.dateColumn)
    candidates = _classification_candidates() if request.task == "classification" else _regression_candidates()
    comparison, best = [], None
    for name, estimator in candidates:
        scores = []
        for a, b in folds:
            if request.task == "classification" and ytrain.iloc[a].nunique() < 2:
                raise ValueError("A selection fold has fewer than two classes. Choose a different split/scope.")
            p = Pipeline([("preprocessor", FoldPreprocessor(request.target, excludes, request.features)), ("model", clone(estimator))])
            p.fit(Xtrain.iloc[a], ytrain.iloc[a])
            predicted = p.predict(Xtrain.iloc[b])
            score = f1_score(ytrain.iloc[b], predicted, average="macro", zero_division=0) if request.task == "classification" else -np.sqrt(mean_squared_error(ytrain.iloc[b], predicted))
            scores.append(float(score))
        mean = float(np.mean(scores))
        comparison.append({"label": name, "value": mean if request.task == "classification" else -mean})
        if best is None or mean > best[0]:
            best = (mean, name, estimator)
    pipeline = Pipeline([("preprocessor", FoldPreprocessor(request.target, excludes, request.features)), ("model", clone(best[2]))])
    pipeline.fit(Xtrain, ytrain)
    prediction = pipeline.predict(Xtest)  # Final evaluation happens only here.
    selection_metric = "Macro F1" if request.task == "classification" else "RMSE"
    visuals = [bar("model-selection", "Training-fold model comparison", comparison, f"{len(folds)} training-only folds; {selection_metric}. Final test scores were not used to select a model.")]
    if request.task == "classification":
        labels = sorted(set(ytest) | set(prediction))
        cm = confusion_matrix(ytest, prediction, labels=labels)
        values = [{"actual": a, "predicted": b, "count": int(cm[i, j])} for i, a in enumerate(labels) for j, b in enumerate(labels)]
        visuals.insert(0, visual("confusion", "Holdout confusion matrix", values, "rect", {
            "x": {"field": "predicted", "type": "nominal", "title": "Predicted class"},
            "y": {"field": "actual", "type": "nominal", "title": "Actual class"},
            "color": {"field": "count", "type": "quantitative", "scale": {"scheme": "blues"}},
            "tooltip": [{"field": "actual"}, {"field": "predicted"}, {"field": "count"}],
        }, f"Observed labels in {len(ytest):,} independent evaluation rows. Cell counts are actual predictions."))
        metrics = [{"label": "Accuracy", "value": float(accuracy_score(ytest, prediction))}, {"label": "Macro F1", "value": float(f1_score(ytest, prediction, average="macro", zero_division=0))}]
        interpretation = f"{best[1]} was selected using training-only {selection_metric}. Accuracy is {metrics[0]['value']:.3f} and macro F1 is {metrics[1]['value']:.3f} on the final evaluation set. Inspect off-diagonal cells for errors by class."
    else:
        metrics = [{"label": "MAE", "value": float(mean_absolute_error(ytest, prediction))}, {"label": "RMSE", "value": float(np.sqrt(mean_squared_error(ytest, prediction)))}, {"label": "R²", "value": float(r2_score(ytest, prediction))}]
        values = [{"actual": float(a), "predicted": float(b), "residual": float(a-b)} for a, b in zip(ytest, prediction)]
        visuals.insert(0, visual("actual-predicted", "Actual versus predicted", values, {"type": "point", "opacity": .55}, {
            "x": {"field": "actual", "type": "quantitative", "title": "Actual value"},
            "y": {"field": "predicted", "type": "quantitative", "title": "Predicted value"},
            "tooltip": [{"field": "actual"}, {"field": "predicted"}, {"field": "residual"}],
        }, f"All {len(values):,} independent evaluation rows; target units are unchanged."))
        visuals.append(visual("residuals", "Prediction residuals", values, {"type": "point", "opacity": .55}, {"x": {"field": "predicted", "type": "quantitative"}, "y": {"field": "residual", "type": "quantitative"}}, "Residual = actual minus predicted; inspect systematic error and changing spread."))
        interpretation = f"{best[1]} was selected using training-only RMSE. Final holdout RMSE is {metrics[1]['value']:.3f} in target units; R² is {metrics[2]['value']:.3f}. These are evaluation scores, not forecasting guarantees."
    features = pipeline.named_steps["preprocessor"].columns_
    return {
        "summary": [f"Evaluated {request.target} with {best[1]}.", f"{len(Xtrain):,} training rows; {len(Xtest):,} final evaluation rows; {strategy} strategy."],
        "findings": [{"id": "performance", "title": "Independent evaluation", "interpretation": interpretation, "evidenceIds": [visuals[0]["id"]]}, {"id": "selection", "title": "How the model was selected", "interpretation": f"Candidate selection used {len(folds)} folds within training data. Imputation, encoding, scaling and feature selection were fitted in each training fold. The selected model was refitted on training rows before final evaluation.", "evidenceIds": ["model-selection"]}],
        "visuals": visuals, "metrics": metrics,
        "nextSteps": ["Check errors against the business goal and the cost of wrong predictions.", "Validate on independent data before using the model for decisions."],
        "limitations": ["This is a bounded AutoMind-style pipeline using the existing scikit-learn candidate models, not the complete AutoMind paper implementation.", "Available columns may include outcome leakage. Review whether features were known at prediction time."] + pipeline.named_steps["preprocessor"].warnings_,
        "evaluation": {"mode": request.evaluation, "strategy": strategy, "groupColumn": group_column, "trainRows": len(Xtrain), "testRows": len(Xtest), "selectionFolds": len(folds), "selectionMetric": selection_metric, "finalEvaluation": evaluation_label, "features": features, "seed": 42},
    }


def _clustering(request, df, excludes):
    if request.target or request.testData or request.evaluation != "automatic":
        raise ValueError("Clustering does not accept a target or separate evaluation dataset.")
    customer_goal = "customer" in request.goal.lower() and any(w in request.goal.lower() for w in ("segment", "cluster", "group"))
    if customer_goal and "customer" not in request.rowGrain.lower():
        raise ValueError("Customer segmentation requires one row per customer. Select a customer-level view; item/source-row clusters cannot be presented as customers.")
    prep = FoldPreprocessor("", excludes, request.features).fit(df)
    X = prep.transform(df)
    unique_count = len(np.unique(X, axis=0))
    if unique_count < 3:
        raise ValueError("Clustering requires at least three distinct usable feature vectors.")
    candidates = [request.clusters] if request.clusters else range(2, min(6, unique_count - 1, len(df) - 1) + 1)
    best = None
    for k in candidates:
        if k >= unique_count or k >= len(df):
            raise ValueError("Cluster count must be smaller than the number of distinct feature vectors and rows.")
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        score = float(silhouette_score(X, model.labels_, sample_size=min(2000, len(X)), random_state=42))
        if best is None or score > best[0]:
            best = (score, model)
    score, model = best
    sizes = [{"label": f"Group {int(k)+1}", "value": int(v)} for k, v in pd.Series(model.labels_).value_counts().sort_index().items()]
    visuals = [bar("cluster-sizes", "Group sizes", sizes, f"K-means, {model.n_clusters} groups; {len(df):,} rows; {request.rowGrain}. Counts include all analysed rows.")]
    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42).fit(X)
        projection = pca.transform(X)
        shown = np.random.default_rng(42).choice(len(X), min(len(X), 1500), replace=False)
        values = [{"x": float(projection[i, 0]), "y": float(projection[i, 1]), "group": f"Group {model.labels_[i]+1}"} for i in shown]
        visuals.append(visual("projection", "Groups in a two-dimensional projection", values, {"type": "point", "opacity": .5}, {"x": {"field": "x", "type": "quantitative", "title": "Principal component 1"}, "y": {"field": "y", "type": "quantitative", "title": "Principal component 2"}, "color": {"field": "group", "type": "nominal"}}, f"PCA of the imputed, scaled and encoded features; {sum(pca.explained_variance_ratio_):.1%} explained variance. Showing {len(shown):,} of {len(X):,} rows with seeded display sampling; clustering uses all rows."))
    numeric = [c for c in prep.columns_ if pd.api.types.is_numeric_dtype(df[c])][:8]
    profile_description = "Compare groups using the available projection."
    if numeric:
        values = []
        for c in numeric:
            std = df[c].std()
            for k in range(model.n_clusters):
                mean = df.loc[model.labels_ == k, c].mean()
                if pd.notna(mean) and std > 0:
                    values.append({"field": c, "group": f"Group {k+1}", "mean": float(mean), "difference": float((mean-df[c].mean())/std)})
        if values:
            contrasts = []
            for c in numeric:
                observed = [v for v in values if v["field"] == c]
                if len(observed) >= 2:
                    low, high = min(observed, key=lambda v: v["mean"]), max(observed, key=lambda v: v["mean"])
                    contrasts.append((high["difference"] - low["difference"], f"{c}: {low['group']} mean {low['mean']:.3g} versus {high['group']} mean {high['mean']:.3g}"))
            if contrasts:
                profile_description = "Largest differences among the displayed numerical attributes: " + "; ".join(text for _, text in sorted(contrasts, reverse=True)[:2]) + ". These are descriptive contrasts, not causes."
            visuals.append(visual("cluster-profiles", "Numeric group profiles", values, "rect", {"x": {"field": "field", "type": "nominal"}, "y": {"field": "group", "type": "nominal"}, "color": {"field": "difference", "type": "quantitative", "scale": {"scheme": "redblue", "domainMid": 0}}, "tooltip": [{"field": "field"}, {"field": "group"}, {"field": "mean", "type": "quantitative"}, {"field": "difference", "type": "quantitative"}]}, "Up to eight numeric attributes; group means relative to the overall mean, in standard deviations. Missing raw values are excluded from each mean. Descriptive differences do not establish causes."))
    size_description = "; ".join(f"{s['label']}: {s['value']:,} rows ({s['value']/len(df):.1%})" for s in sizes)
    return {
        "summary": [f"Found {model.n_clusters} groups across {len(df):,} rows.", f"Silhouette score: {score:.3f}; features: {', '.join(prep.columns_)}."],
        "findings": [{"id": "groups", "title": "Groups describe this selected population", "interpretation": f"{size_description}. Each row represents {request.rowGrain}. Silhouette ({score:.3f}) measures separation in the encoded feature space; it is not prediction accuracy or proof of meaningful business segments.", "evidenceIds": ["cluster-sizes"]}, {"id": "profiles", "title": "Interpret groups using their observed characteristics", "interpretation": profile_description + " PCA is a lossy display; group membership was computed in the full preprocessed feature space.", "evidenceIds": [v["id"] for v in visuals[1:]]}],
        "visuals": visuals, "metrics": [{"label": "Groups", "value": model.n_clusters}, {"label": "Silhouette", "value": score}],
        "nextSteps": ["Check whether group differences are useful for the analysis goal.", "Check stability on another period or population before naming segments."],
        "limitations": ["K-means uses imputation, scaling and one-hot encoding; this distance may not match domain similarity.", "Auto chooses 2–6 groups by silhouette on this population; no independent stability validation is claimed.", "This uses the existing ML stack and is not the full AutoMind paper implementation."],
        "evaluation": {"features": prep.columns_, "clusters": model.n_clusters, "method": "K-means", "seed": 42},
    }
