"""Local integration checks; uses existing imported datasets, creates real saved runs."""
import copy
import json
import uuid
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

URL = "http://127.0.0.1:3001/api/explorer"

def api(body=None, suffix=""):
    request = Request(URL + suffix, data=json.dumps(body).encode() if body else None,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=240) as response:
            return json.load(response)
    except HTTPError as exc:
        return json.load(exc)

def setup(dataset=None, view=None):
    return api({"action": "analysis.setup", "ref": {"datasetId": dataset, "viewId": view}})

def run_input(source, task, config, goal):
    context = source["context"]
    return {"id": str(uuid.uuid4()), "ref": {k: context[k] for k in ("projectId", "datasetId", "viewId", "schemaVersion")}, "filters": context["filters"], "datasetVersion": source["profile"]["datasetVersion"], "sourceVersion": source["profile"]["sourceVersion"], "task": task, "goal": goal, "config": config, **({"charts": context["charts"]} if task == "exploration" else {})}

def run(input):
    result = api({"action": "analysis.run", "input": input})
    assert result.get("status") == "completed", result.get("error", result)
    assert result["report"]["visuals"]
    return result

def main():
    catalog = api()
    training = next(d["id"] for d in catalog["datasets"] if d["name"] == "Heart training data")
    test = next(d["id"] for d in catalog["datasets"] if d["name"] == "Heart test data")
    customers = next(d["id"] for d in catalog["datasets"] if d["name"] == "Customers")
    education = next(d["id"] for d in catalog["datasets"] if d["name"] == "Edudata_English")
    source = setup(training)
    assert source["profile"]["rowCount"] == 734
    assert source["capabilities"]["classification"]
    base = run_input(source, "classification", {"target": "HeartDisease", "features": None, "evaluation": "automatic", "split": "auto"}, "Predict HeartDisease and evaluate model performance.")
    assert "test" not in base["config"]
    stale = copy.deepcopy(base)
    stale["datasetVersion"] = "expired"
    assert "Data changed" in api({"action": "analysis.run", "input": stale})["error"]
    stale_source = copy.deepcopy(base)
    stale_source["sourceVersion"] = "expired"
    assert "source data changed" in api({"action": "analysis.run", "input": stale_source})["error"]
    missing = copy.deepcopy(base)
    missing["config"]["target"] = None
    assert "Choose a target" in api({"action": "analysis.run", "input": missing})["error"]
    test_scope = setup(test)
    unlabelled = copy.deepcopy(base)
    unlabelled["config"].update(evaluation="separate", test={"ref": test_scope["context"], "filters": test_scope["context"]["filters"], "datasetVersion": test_scope["profile"]["datasetVersion"], "sourceVersion": test_scope["profile"]["sourceVersion"]})
    assert "cannot measure model quality" in api({"action": "analysis.run", "input": unlabelled})["error"]
    classification = run(base)
    same = run(base)
    assert same["id"] == classification["id"] and same["report"]["scope"]["snapshotId"] == classification["report"]["scope"]["snapshotId"]
    conflict = copy.deepcopy(base)
    conflict["goal"] = "Different request"
    assert "another request" in api({"action": "analysis.run", "input": conflict})["error"]
    saved = api({"action": "analysis.record", "id": classification["id"]})
    assert saved["report"] == classification["report"]
    restored = api({"action": "analysis.setup", "fromRun": classification["id"]})
    assert restored["drafts"]["classification"]["target"] == "HeartDisease"
    assert restored["restoredTask"] == "classification"
    education_source = setup(education)
    target = 'I am willing to share my digital skills with other students'
    assert education_source['profile']['rowCount'] == 1305
    regression = run(run_input(education_source, "regression", {"target": target, "evaluation": "automatic", "split": "auto"}, "Predict willingness to share digital skills with other students on the recorded 1–5 survey scale, and explain prediction errors."))
    assert regression['input']['config']['target'] == target
    assert 'Timestamp' not in regression['report']['evaluation']['features']
    assert regression['report']['evaluation']['strategy'] == 'group'
    assert 'fractional prediction' in ' '.join(regression['report']['limitations'])
    customer_source = setup(customers)
    assert customer_source["profile"]["rowCount"] == 2000
    assert next(f for f in customer_source["context"]["fields"] if f["name"] == "CustomerID")["role"] == "identifier"
    clustering = run(run_input(customer_source, "clustering", {}, "Segment customers into groups with similar characteristics and explain their differences."))
    assert clustering["report"]["scope"]["rowCount"] == 2000
    boxes = [v for v in clustering['report']['visuals'] if isinstance(v['chartSpec']['mark'], dict) and v['chartSpec']['mark']['type'] == 'boxplot']
    assert len(boxes) >= 3
    assert all(len(v['chartSpec']['data']['values']) == 2000 for v in boxes)
    sales = setup(f"project:{catalog['projectId']}", "sales")
    exploration = run(run_input(sales, "exploration", {"includeCharts": True}, "Explore item sales over time and compare order volumes across customer states."))
    assert exploration["report"]["scope"]["rowCount"] == sales["profile"]["rowCount"]
    assert abs(exploration["report"]["metrics"][2]["value"] - sales["profile"]["metrics"]["itemSales"]) < 0.01
    for record in (exploration, clustering, classification, regression):
        report = record['report']
        assert len(report['insights']) >= 3
        visual_ids = {v['id'] for v in report['visuals']}
        assert all(set(i['evidenceIds']) <= visual_ids for i in report['insights'])
        narrative = ' '.join(report['summary'] + report['limitations'] + [f['interpretation'] for f in report['findings']])
        assert all(term not in narrative for term in ('SUM(', 'LLM', 'not the full AutoMind paper', 'existing ML stack'))
    result = {task: {"id": record["id"], "metrics": record["report"]["metrics"], "evaluation": record["report"].get("evaluation"), "rowCount": record["report"]["scope"]["rowCount"]} for task, record in [("exploration", exploration), ("classification", classification), ("regression", regression), ("clustering", clustering)]}
    output = Path(".local/analysis-check.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("PASS: four real tasks; grounded insights; original-unit boxplots; chosen survey target; stale data, missing target, unlabelled test, idempotence, immutable report and rerun configuration.")

if __name__ == "__main__":
    main()
