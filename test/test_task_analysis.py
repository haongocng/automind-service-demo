import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.task_analysis import TaskRequest, FoldPreprocessor, _split, analyze_task


def request(task="classification", **kwargs):
    rng = np.random.default_rng(7)
    data = [{"row_id": i, "x": float(rng.normal()), "Outcome": i % 2} for i in range(120)]
    base = dict(task=task, goal="Predict Outcome and evaluate performance.", data=data,
                fields=[{"name": "row_id", "role": "identifier"}, {"name": "x", "role": "number"}, {"name": "Outcome", "role": "number"}],
                target="Outcome", rowGrain="One source row")
    base.update(kwargs)
    return TaskRequest(**base)


class AnalysisEvaluationTests(unittest.TestCase):
    def test_final_holdout_never_used_for_candidate_selection(self):
        req = request()
        observed = []
        class Spy(DecisionTreeClassifier):
            def fit(self, X, y, **kw):
                observed.append(len(X))
                return super().fit(X, y, **kw)
        with patch("app.services.task_analysis._classification_candidates", return_value=[("Spy", Spy(random_state=42))]):
            result = analyze_task(req)
        self.assertEqual(observed, [60, 60, 60, 90])
        self.assertEqual(result["evaluation"]["testRows"], 30)
        self.assertEqual(sum(v["count"] for v in result["visuals"][0]["chartSpec"]["data"]["values"]), 30)

    def test_preprocessor_is_fit_only_on_training_values(self):
        p = FoldPreprocessor("", (), ["x"]).fit(pd.DataFrame({"x": [1., 2., 3.]}))
        p.transform(pd.DataFrame({"x": [1000.]}))
        self.assertEqual(p.preprocessor_.named_transformers_["numeric"].named_steps["imputer"].statistics_[0], 2.)
        self.assertEqual(p.preprocessor_.named_transformers_["numeric"].named_steps["scaler"].mean_[0], 2.)

    def test_repeated_entities_are_disjoint(self):
        req = request(data=[{"row_id": i // 3, "x": i, "Outcome": i % 2} for i in range(120)])
        frame = pd.DataFrame(req.data)
        train, test, strategy, _, _ = _split(req, frame, frame.Outcome)
        self.assertEqual(strategy, "group")
        self.assertFalse(set(frame.iloc[train].row_id) & set(frame.iloc[test].row_id))

    def test_time_split_keeps_equal_dates_on_same_side(self):
        req = request(split="time", dateColumn="date", data=[{"row_id": i, "x": i, "Outcome": i % 2, "date": f"2024-01-{i // 6 + 1:02d}"} for i in range(120)])
        frame = pd.DataFrame(req.data)
        train, test, strategy, _, _ = _split(req, frame, frame.Outcome)
        self.assertEqual(strategy, "time")
        self.assertLess(frame.iloc[train].date.max(), frame.iloc[test].date.min())

    def test_unlabelled_test_cannot_produce_quality_metrics(self):
        req = request(evaluation="separate", testData=[{"row_id": 500+i, "x": float(i)} for i in range(30)])
        with self.assertRaisesRegex(ValueError, "target column"):
            analyze_task(req)

    def test_duplicate_separate_population_is_rejected(self):
        req = request()
        req.evaluation = "separate"
        req.testData = req.data[:30]
        with self.assertRaisesRegex(ValueError, "duplicate rows"):
            analyze_task(req)

    def test_regression_reports_numeric_errors(self):
        req = request("regression", data=[{"row_id": i, "x": i*.9, "Outcome": float(i)+.5} for i in range(120)])
        with patch("app.services.task_analysis._regression_candidates", return_value=[("Tree", DecisionTreeRegressor(random_state=42))]):
            result = analyze_task(req)
        self.assertEqual([m["label"] for m in result["metrics"]], ["MAE", "RMSE", "R²"])
        self.assertEqual(len(result["visuals"][0]["chartSpec"]["data"]["values"]), 30)

    def test_clustering_is_real_and_respects_grain(self):
        req = request("clustering", target=None, goal="Discover similar groups.")
        result = analyze_task(req)
        self.assertEqual(sum(v["value"] for v in result["visuals"][0]["chartSpec"]["data"]["values"]), 120)
        self.assertNotIn("Accuracy", [m["label"] for m in result["metrics"]])
        self.assertNotIn("row_id", result["evaluation"]["features"])
        sizes = {v["label"]: v["value"] for v in result["visuals"][0]["chartSpec"]["data"]["values"]}
        profiles = next(v for v in result["visuals"] if v["id"] == "cluster-profiles")["chartSpec"]["data"]["values"]
        for field in {v["field"] for v in profiles}:
            weighted_mean = sum(v["mean"] * sizes[v["group"]] for v in profiles if v["field"] == field) / len(req.data)
            self.assertAlmostEqual(weighted_mean, np.mean([row[field] for row in req.data]))
        req.goal = "Segment customers into similar groups."
        with self.assertRaisesRegex(ValueError, "one row per customer"):
            analyze_task(req)


if __name__ == "__main__":
    unittest.main()
