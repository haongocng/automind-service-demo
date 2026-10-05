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
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
bash scripts/start_backend.sh
```

Open API docs:

```text
http://localhost:8000/docs
```

## Recovered WrenAI frontend

The frontend lives in a separate repository:
[haongocng/WrenAI, branch automind-prediction-demo](https://github.com/haongocng/WrenAI/tree/automind-prediction-demo).
The recovered checkout is in `WrenAI/`, with the Next.js frontend in `WrenAI/wren-ui/`.

After the local dependencies have been installed, run these commands from this
repository in **two separate terminals**:

```bash
# Terminal 1: Python backend, using venv; LLM insights disabled by default
bash scripts/start_backend.sh
```

```bash
# Terminal 2: WrenAI frontend, using the local Node 20 runtime
bash scripts/start_frontend.sh
```

For a demo using an existing production build, start the frontend with
`WREN_UI_MODE=production bash scripts/start_frontend.sh`. The default remains
development mode; rebuild the frontend after code changes before using production mode.

Open **http://127.0.0.1:3001/**. The frontend script defaults to **data-only mode**:
it caches the nine original WrenAI E-commerce Parquet files, then uses the
existing native DuckDB library with two threads and a 512 MiB database memory
limit. No Docker, Ollama or LLM API key is needed for this mode. Node/Next.js
and AutoMind consume additional memory beyond that DuckDB limit.

The workspace uses **AutoMind** branding with a connected-A symbol and matching
browser favicon. **Guidance** in the header opens
**http://127.0.0.1:3001/guidance**: a short getting-started guide and four examples
linked to the available E-commerce, Customers, Heart and Edudata_English datasets.

On first use, select **E-commerce** under **Play around with sample data**.
WrenAI saves the models and relationships, then opens **Home / Data explorer**
with real data, a preview table and initial charts. The recovered local project contains
9 models, 9 relationships and 99,441 orders; it was initialized using the
sample selected by the user, not restored from the inaccessible server.

Use **Back to datasets** (http://127.0.0.1:3001/home/datasets) to choose the
connected source, explore an individual model, or import a local CSV. Import
opens the explorer immediately and preserves the connected E-commerce source.
Filters, saved chart definitions, Rows/Fields and pagination use real queries.
**Analyze with AutoMind** saves the Explorer state and opens the full-page
**Analysis setup**. Choose Data exploration, Clustering, Classification or
Regression; only **Run analysis** starts work. Results open in a wide
**Analysis report** and remain in **Saved analyses**, with frozen evidence.
Supervised tasks default to an automatic split and require a verified target;
models are selected using training-only folds before final holdout evaluation.
ML tasks process the entire selected scope, up to 20,000 rows. Exploration
uses full-population SQL aggregates. No Ollama or LLM API is required for these
four tasks. This is a bounded AutoMind-style integration, not the complete
paper implementation. See [Analysis implementation and screenshots](docs/ANALYSIS_IMPLEMENTATION.md).

Reports now lead with **Key insights**, link explanations to their visual evidence,
and describe data-specific limitations. Customers includes original-unit boxplots;
an unlabelled test selection offers **Use automatic split**. The imported
**Edudata_English** regression draft uses the user-selected target
`I am willing to share my digital skills with other students`.
See [report updates, verified results and screenshots](docs/REPORT_INSIGHTS_UPDATE.md).

The existing ML demos remain at
**http://127.0.0.1:3001/automind-prediction**. **Run Prediction from WrenAI Data**
queries 1,000 sample records for the existing prediction task.
AI chat, AI deployment/indexing and LLM-generated commentary are disabled in
this mode. Advanced MDL calculated fields and implicit relationship traversal
require the regular Wren Engine; basic model/column aliases and explicit SQL
joins work locally.

All navigation pages remain accessible in data-only mode: Data explorer, Data chat, Dashboard,
Knowledge (question-SQL pairs and instructions), API history, Modeling and
Guidance. Knowledge forms can be opened and their SQL preview works locally.
AI chat, question generation and saving/indexing Knowledge require the AI
service; the affected controls show that requirement instead of blocking pages.

The frontend script applies SQLite migrations before starting Next.js. Metadata
is persisted in `WrenAI/wren-ui/db.sqlite3`; source data is persisted in
`.local/wren-data/wren.duckdb`. Existing downloads are reused on restart.
Press `Ctrl+C` in each terminal to stop.

See [the explorer implementation and verification guide](docs/EXPLORER_IMPLEMENTATION.md)
for changed files, analysis limitations, CSV details and screenshots.

For a configured full WrenAI stack, use
`WREN_DATA_ONLY_MODE=false bash scripts/start_frontend.sh`. That mode uses the
regular remote Engine/AI/Ibis services and a configured LLM/embedding provider.
Ollama is one optional provider. See `docs/WRENAI_RUN_GUIDE.md` for the two modes.

The recovered GitHub branch includes the E-commerce and Heart Disease demo
interface. The **Custom Prediction / CSV upload** form visible in the thesis
screenshots is absent from this branch and its available commit history; its
original frontend source has not been recovered. The current AutoMind backend
does already expose `POST /predict/upload`.

For a fresh machine, first complete the Python setup above, then restore the
frontend dependencies with:

```bash
git clone --single-branch --branch automind-prediction-demo https://github.com/haongocng/WrenAI.git WrenAI
git -C WrenAI apply ../scripts/wrenai-data-only.patch
npm install --prefix .local/node-runtime --cache .local/npm-cache --no-audit --no-fund --package-lock=false node@20.20.2
cd WrenAI/wren-ui
export PATH="$PWD/../../.local/node-runtime/node_modules/node/bin:$PATH"
export YARN_ENABLE_GLOBAL_CACHE=false
export YARN_CACHE_FOLDER="$PWD/../../.local/yarn-cache"
export YARN_GLOBAL_FOLDER="$PWD/../../.local/yarn-global"
node .yarn/releases/yarn-4.5.3.cjs install --immutable
```

`WrenAI/` is an independent Git checkout and `.local/` contains the runtime and
package caches; both are ignored by this backend repository. The local frontend
changes, including the explorer, are preserved in `scripts/wrenai-data-only.patch`
so a fresh clone can reproduce this implementation. Do not apply that patch
twice to this checkout. Local databases and imported CSV datasets are separate
from the source patch; back up both database files to preserve them.

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

Prepared dataset storage:

```text
examples/heart_disease/heart_disease.sqlite
```

The SQLite database contains two tables:

```text
heart_train
heart_test
```

The source CSV files are kept as fallback inputs:

```text
examples/heart_disease/heart_train.csv
examples/heart_disease/heart_test.csv
```

Regenerate the SQLite database from CSV files with:

```bash
./venv/bin/python scripts/create_heart_disease_sqlite.py
```

The training table contains labeled rows with target column:

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



## Lightweight Multi-Agent Architecture

The public `SimplifiedAutoMindPipeline` interface is kept for compatibility, but it now delegates to `AutoMindOrchestrator` in `app/agents/orchestrator.py`. The orchestrator executes small agents in order:

```text
DataProfilerAgent
PreprocessingAgent
ModelingAgent
ModelAuditAgent
ReportAgent
InsightAgent
```

Each response includes a non-breaking trace field:

```text
agent_trace
```

The structured report also includes:

```text
report.agent_workflow
```

These fields are for traceability and do not replace existing response fields. Existing report sections, legacy fields, rule-based insights, and optional DeepInfra `report.agent_insights` remain available.

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
agent_trace
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

The recovered page is `WrenAI/wren-ui/src/pages/automind-prediction.tsx`.
It calls Next.js API routes in `src/pages/api/automind/`, which proxy requests to
this FastAPI backend. The chart renderer is
`src/components/pages/automind/AutoMindChart.tsx` and uses Vega.

The interface renders an Insight Report first, then expandable Detailed Agent
Reports. It uses `report.eda.charts` for EDA,
`report.prediction_results.charts` for feature importance and prediction charts,
and `report.model_audit.charts` for model audit charts. It also displays sample
predictions, metrics, the agent execution trace and the Markdown report.
