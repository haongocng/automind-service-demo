import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from app.services.exploration_analysis import ExplorationRequest, analyze_exploration
from app.services.task_analysis import _split, analyze_task
from app.services.preprocessing import prepare_features
from test_task_analysis import request


class ReportInsightsTests(unittest.TestCase):
    def test_string_timestamp_is_not_encoded_as_a_predictor(self):
        frame = pd.DataFrame({'Timestamp': pd.Series([str(i) for i in range(120)], dtype='str'), 'x': np.arange(120), 'Outcome': np.arange(120) % 5})
        prepared = prepare_features(frame, 'Outcome', [])
        self.assertNotIn('Timestamp', prepared.X.columns)
        self.assertIn('Timestamp', prepared.dropped_columns)

    def test_sales_share_and_comparable_months_use_correct_scope(self):
        result = analyze_exploration(ExplorationRequest(
            snapshotId='saved', goal='Review sales', method='Frozen snapshot',
            context={'datasetName': 'Sales', 'viewName': 'Sales overview', 'viewId': 'sales', 'rowGrain': 'One order item', 'filters': {'values': {'order_status': 'delivered'}}},
            profile={'rowCount': 12, 'dateMin': '2017-01-15', 'dateMax': '2018-03-05', 'fields': [{'name': 'item_price', 'nullCount': 0}], 'metrics': {'distinctOrders': 10, 'itemSales': 1000}},
            evidence=[
                {'title': 'Sales over time', 'definition': 'sum(item_price)', 'scope': 'Current filters', 'metricLabel': 'Item sales', 'chart': {'kind': 'time', 'timeGrain': 'month'}, 'values': [{'label': '2017-01-01', 'value': 10}, {'label': '2018-01-01', 'value': 100}, {'label': '2017-02-01', 'value': 100}, {'label': '2018-02-01', 'value': 150}, {'label': '2017-03-01', 'value': 20}, {'label': '2018-03-01', 'value': 100}]},
                {'title': 'Orders by state', 'definition': 'count_distinct(order_id)', 'scope': 'Current filters', 'metricLabel': 'Distinct orders', 'chart': {'groupBy': 'customer_state', 'measure': 'order_id', 'aggregation': 'count_distinct'}, 'values': [{'label': 'SP', 'value': 4}, {'label': 'RJ', 'value': 3}]},
            ]))
        by_id = {v['id']: v for v in result['insights']}
        self.assertIn('February 2018', by_id['comparable-0']['detail'])
        self.assertIn('50.0%', by_id['comparable-0']['detail'])
        self.assertIn('40.0%', by_id['group-1']['detail'])
        self.assertIn('70.0%', by_id['group-1']['detail'])
        text = ' '.join(result['summary'] + result['limitations'] + [v['detail'] for v in result['insights']])
        self.assertNotIn('SUM(', text)
        self.assertNotIn('LLM', text)
        self.assertIn('delivered orders only', text)

    def test_repeated_answers_at_different_timestamps_do_not_leak(self):
        rows = [{'row_id': i, 'Timestamp': str(i), 'x': i // 2, 'Outcome': float(i // 2)} for i in range(120)]
        req = request('regression', data=rows, fields=[{'name': 'row_id', 'role': 'identifier'}, {'name': 'Timestamp', 'role': 'category'}, {'name': 'x', 'role': 'number'}, {'name': 'Outcome', 'role': 'number'}])
        frame = pd.DataFrame(rows)
        train, test, strategy, groups, _ = _split(req, frame, frame.Outcome)
        self.assertEqual(strategy, 'group')
        self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))
        self.assertFalse(set(frame.iloc[train].x) & set(frame.iloc[test].x))

    def test_boxplots_store_every_original_observation(self):
        req = request('clustering', target=None, goal='Discover groups.', data=[{'row_id': i, 'x': float(i % 10 + (100 if i >= 60 else 0)), 'Outcome': i % 2} for i in range(120)])
        result = analyze_task(req)
        boxes = [v for v in result['visuals'] if isinstance(v['chartSpec']['mark'], dict) and v['chartSpec']['mark']['type'] == 'boxplot']
        self.assertTrue(boxes)
        for box in boxes:
            field = box['chartSpec']['encoding']['y']['title']
            values = box['chartSpec']['data']['values']
            self.assertEqual(sorted(v['value'] for v in values), sorted(row[field] for row in req.data))
            for size in result['visuals'][0]['chartSpec']['data']['values']:
                self.assertEqual(sum(v['group'] == size['label'] for v in values), size['value'])
        self.assertGreaterEqual(len(result['insights']), 3)

    def test_ordinal_regression_keeps_population_and_holdout_separate(self):
        req = request('regression', data=[{'row_id': i, 'x': float(i), 'Outcome': i % 5 + 1} for i in range(120)])
        with patch('app.services.task_analysis._regression_candidates', return_value=[('Tree', DecisionTreeRegressor(random_state=42))]):
            result = analyze_task(req)
        visuals = {v['id']: v for v in result['visuals']}
        self.assertEqual(sum(v['value'] for v in visuals['population-target']['chartSpec']['data']['values']), 120)
        self.assertEqual(sum(v['value'] for v in visuals['target-distribution']['chartSpec']['data']['values']), 30)
        self.assertIn('fractional prediction', ' '.join(result['limitations']))
        self.assertIn('training mean', ' '.join(v['detail'] for v in result['insights']))
        residual_finding = next(f for f in result['findings'] if f['id'] == 'residual-pattern')
        self.assertIn('Mean predicted-minus-observed difference', residual_finding['interpretation'])
        self.assertTrue(all(set(v['evidenceIds']) <= set(visuals) for v in result['insights']))


if __name__ == '__main__':
    unittest.main()
