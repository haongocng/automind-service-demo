# AutoMind-service

Lightweight FastAPI service for WrenAI + AutoMind prediction demos.

Version 1 implements one simplified, reusable AutoMind-style pipeline:

1. Data profiling
2. Target transformation
3. Preprocessing
4. Model selection and training
5. Model auditing
6. Feature importance
7. Rule-based insight synthesis

The demo endpoint predicts whether an e-commerce order receives a good review:

```text
good_review = 1 if review_score >= 4 else 0
```

The core pipeline is generic and now also includes a prepared Heart Disease classification demo through `POST /predict/heart-disease`.

## Setup

```bash
python3 -m venv avenv
source avenv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./avenv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Open API docs:

```text
http://localhost:8000/docs
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Run E-commerce Good Review Demo

With built-in synthetic demo data:

```bash
curl -X POST http://localhost:8000/predict/ecommerce-good-review
```

With a request body:

```bash
curl -X POST http://localhost:8000/predict/ecommerce-good-review \
  -H "Content-Type: application/json" \
  -d @examples/ecommerce_good_review_request.json
```


## Run Heart Disease Classification Demo

Prepared dataset files:

```text
examples/heart_disease/heart_train.csv
examples/heart_disease/heart_test.csv
```

The training file contains labeled rows with target column:

```text
HeartDisease
```

Run the prepared demo:

```bash
curl -X POST http://localhost:8000/predict/heart-disease \
  -H "Content-Type: application/json" \
  -d '{}'
```

This endpoint performs binary classification:

```text
0 = no heart disease
1 = heart disease
```

Medical disclaimer: this workflow is for demonstration and research only. The output is not medical advice, diagnosis, treatment guidance, or a clinical decision system. Real clinical deployment would require expert validation, calibration, bias assessment, and regulatory review.

For now, `heart_test.csv` is loaded for availability checks, but prediction-only output for unlabeled test rows is deferred until the pipeline exposes a fitted model safely.

## Generic Prediction Endpoint

Use `POST /predict` for future domains such as HeartDisease:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/heart_disease_request.json
```

The generic endpoint expects enough row-level records to train/test a model. Very small examples are included only to show request shape.

## Demo Response Shape

```json
{
  "status": "success",
  "summary": {
    "task": "Good review prediction",
    "rows": 96,
    "target": "good_review",
    "selected_model": "RandomForestClassifier"
  },
  "metrics": {
    "accuracy": 0.84,
    "precision": 0.82,
    "recall": 0.79,
    "f1": 0.8,
    "confusion_matrix": [[4, 2], [2, 16]]
  },
  "charts": {
    "feature_importance": [],
    "class_distribution": [],
    "prediction_distribution": []
  },
  "insight": "Business-readable insight text.",
  "warnings": []
}
```

## Notes

- Version 1 does not connect directly to a database.
- Raw records are never sent to an LLM prompt.
- The frontend should render charts from the JSON under `charts`.

## Optional LLM InsightAgent

AutoMind-service can optionally enrich the structured report with a lightweight LLM InsightAgent. This is disabled by default and the service continues to work with rule-based insights when the LLM is unavailable.

Create a local `.env` file in the AutoMind-service project root:

```text
/home/haonn/wrenai-demo/AutoMind-service/.env
```

Example DeepInfra configuration:

```bash
LLM_INSIGHT_ENABLED=true
LLM_PROVIDER=deepinfra
LLM_API_BASE=https://api.deepinfra.com/v1/openai
LLM_MODEL=deepseek-ai/DeepSeek-V3
LLM_TIMEOUT_SECONDS=30
LLM_DEBUG=false
DEEP_INFRA_API_KEY=<your_deepinfra_api_key>
```

Supported API key environment variables, in lookup order:

```text
LLM_API_KEY
DEEP_INFRA_API_KEY
DEEPINFRA_API_KEY
```

The local `.env` loader uses only Python stdlib and does not override environment variables that are already set. `.env` is ignored by git and must not be committed.

Set `LLM_DEBUG=true` only while debugging. It adds a safe fallback reason to warnings, such as `invalid_json`, `http_429`, `missing_api_key`, or `invalid_shape`. It never exposes API keys, raw prompts, raw LLM responses, or stack traces.

The InsightAgent sends only a compact report summary to the LLM:

```text
dataset_overview
eda_summary
metrics
top_features
warnings
limitations
```

It does not send raw records, full dataframes, or row-level prediction data. If the LLM is disabled, missing an API key, times out, or returns invalid JSON, AutoMind-service falls back to the existing rule-based report. When LLM insight succeeds, the response includes:

```text
report.agent_insights
```


## AutoMind-style Report Response

Version 2 adds a structured `report` object while keeping the older top-level fields for backward compatibility.

Important response sections:

```text
report.title
report.executive_summary
report.dataset_overview
report.eda.summary
report.eda.charts
report.key_insights
report.prediction_task
report.prediction_results.sample_predictions
report.prediction_results.charts
report.model_audit.metrics
report.model_audit.charts
report.recommendations
report.warnings
report.limitations
report.agent_insights
report.report_markdown
```

The older fields still exist:

```text
summary
metrics
charts
insight
warnings
legacy
```

## Chart-ready JSON

The backend does not render charts directly. It returns chart specifications that the frontend can render later.

Each chart object follows this shape:

```json
{
  "id": "feature_importance",
  "title": "Top Feature Importance",
  "type": "bar",
  "description": "Most influential features used by the selected model.",
  "data": [
    {"feature": "delivery_days", "value": 0.16}
  ],
  "x": "feature",
  "y": "value"
}
```

Current report chart groups:

```text
report.eda.charts
report.prediction_results.charts
report.model_audit.charts
```

Minimum returned chart specs:

```text
class_distribution
missing_values
numeric_columns
prediction_distribution
feature_importance
```

## Sample Predictions

`report.prediction_results.sample_predictions` returns at most 20 validation rows.

For the e-commerce good review task, each item may include:

```json
{
  "row_index": 10,
  "order_id": "demo_order_011",
  "actual": "Good review",
  "predicted": "Good review",
  "probability_good_review": 0.87
}
```

The service does not return all row-level predictions by default.

## Example Response

A generated example response is available at:

```text
examples/ecommerce_good_review_response_example.json
```

## Frontend Rendering Notes

FastAPI Swagger UI only displays JSON. It will not render charts visually.

A future WrenAI frontend page should:

1. Call `POST /predict/ecommerce-good-review`
2. Render `report.model_audit.charts[0]` as feature importance
3. Render `report.eda.charts` for EDA charts
4. Render `report.prediction_results.charts` for prediction distribution
5. Show `report.prediction_results.sample_predictions` as a compact table
6. Show `report.report_markdown` or render the structured report sections directly
