"""Read-only smoke checks of the running explorer, real charts, and page routes.

Does not create prediction runs, change user preferences, or import test files.
Run: venv/bin/python scripts/check_explorer.py
"""
import json
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:3001"


def post(payload):
    with urlopen(Request(BASE + "/api/explorer", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}), timeout=90) as response:
        return json.load(response)


def main():
    data = post({"action": "context", "ref": {}})
    ctx = data["context"]
    filters = ctx["defaultFilters"]
    assert ctx["viewId"] == "sales", "Select the connected E-commerce dataset for this smoke test"
    assert ctx["modelCount"] == 9
    preview = post({"action": "rows", "ref": ctx, "filters": filters, "page": 1, "size": 5})
    assert len(preview["rows"]) == 5
    assert preview["total"] > 100000
    assert all(r["order_status"] == "delivered" for r in preview["rows"])
    print(f"PASS: {preview['total']:,} delivered items, real IDs/prices and pagination")
    charts = [post({"action": "chart", "ref": ctx, "filters": filters, "chart": c}) for c in ctx["charts"]]
    assert all(c["values"] for c in charts)
    assert charts[1]["values"][0]["label"] == "SP"
    scoped = {"values": {"customer_state": "RJ", "order_status": "delivered"}}
    region = post({"action": "chart", "ref": ctx, "filters": scoped, "chart": ctx["charts"][1]})
    assert len(region["values"]) == 1 and region["values"][0]["label"] == "RJ"
    assert region["values"][0]["value"] == next(r["value"] for r in charts[1]["values"] if r["label"] == "RJ")
    empty = post({"action": "rows", "ref": ctx, "filters": {"values": {"customer_state": "__NO_MATCH__"}}, "page": 1, "size": 5})
    assert empty["total"] == 0 and empty["rows"] == []
    print("PASS: chart filters agree, empty scope returns no fabricated values")
    with urlopen(BASE + "/api/explorer", timeout=90) as response:
        catalog = json.load(response)
    for name, count, columns in [("Heart training data", 734, 12), ("Heart test data", 184, 11)]:
        dataset = next((d for d in catalog["datasets"] if d["name"] == name), None)
        if not dataset:
            continue
        loaded = post({"action": "context", "ref": {"datasetId": dataset["id"]}})
        csvctx = loaded["context"]
        assert loaded["profile"]["rowCount"] == count
        assert len(csvctx["fields"]) == columns and not csvctx.get("dateField")
        assert csvctx["rowGrain"] == "Not defined"
        assert not csvctx["defaultFilters"].get("values")
        csvrows = post({"action": "rows", "ref": csvctx, "filters": csvctx["filters"], "page": 1, "size": 5})
        assert csvrows["total"] == count and len(csvrows["rows"]) == 5
        for chart in csvctx["charts"]:
            assert post({"action": "chart", "ref": csvctx, "filters": csvctx["filters"], "chart": chart})["values"]
        print(f"PASS CSV: {name}, {count} rows / {columns} fields, real preview and generic charts")
    for route in ["/home", "/home/chat", "/home/datasets", "/home/analyses", "/modeling", "/knowledge/question-sql-pairs", "/api-management/history"]:
        with urlopen(BASE + route, timeout=90) as response:
            assert response.status == 200
        print("PASS route:", route)
    print(json.dumps({"sampleProfile": data["profile"], "charts": charts}, indent=2)[:1000])


if __name__ == "__main__":
    main()
