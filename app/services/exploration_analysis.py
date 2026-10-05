"""Descriptive analysis of full-population SQL evidence, without an LLM or training.

The Wren adapter freezes the data and computes aggregates. No preview records or
fabricated findings are used. This is a descriptive task alongside prediction,
not an implementation of the paper's complete autonomous research algorithm.
"""
from typing import Any
from calendar import monthrange
from datetime import date, datetime
from pydantic import BaseModel, Field
from app.services.report_insights import insight, number


class ExplorationRequest(BaseModel):
    snapshotId: str = Field(min_length=1, max_length=100)
    goal: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any]
    profile: dict[str, Any]
    evidence: list[dict[str, Any]] = Field(max_length=10)
    method: str


def _period(label, grain):
    try:
        parsed = datetime.fromisoformat(str(label)[:10])
        return parsed.strftime('%B %Y') if grain == 'month' else parsed.strftime('%Y') if grain == 'year' else parsed.strftime('%d %b %Y')
    except ValueError:
        return str(label)


def analyze_exploration(request: ExplorationRequest) -> dict[str, Any]:
    profile = request.profile
    context = request.context
    count = int(profile["rowCount"])
    if count < 0:
        raise ValueError("Row count cannot be negative")
    summary = [f"This overview covers {count:,} records in {context['datasetName']} / {context['viewName']}. Each record represents {context['rowGrain'].lower()}; the saved filters define the population described below."]
    insights, findings, limits = [], [], []
    if profile.get("dateMin") and profile.get("dateMax"):
        summary.append(f"Filtered date coverage: {str(profile['dateMin'])[:10]} to {str(profile['dateMax'])[:10]}.")
    metrics = profile.get("metrics")
    if metrics:
        amount = "Not available" if metrics["itemSales"] is None else f"{metrics['itemSales']:,.2f}"
        orders = metrics['distinctOrders']
        summary.append(f"Selected item sales total {amount} price units across {orders:,} distinct orders. Freight is excluded; each order is counted once even when it contains several items.")
        if orders and metrics['itemSales'] is not None:
            insights.append(insight('sales-scale', 'The scale of sales in this scope', f"The selected orders contain an average of {count/orders:.2f} items and {number(metrics['itemSales']/orders)} price units per order. Compare this with order volume when reviewing changes: more item sales can reflect more orders, more items per order, or a different price mix."))
    if count == 0:
        summary.append("No rows match this scope. No trend or ranking can be inferred.")
    evidence = []
    for i, item in enumerate(request.evidence):
        values = item.get("values", [])
        valid = [r for r in values if r.get("value") is not None]
        ident = f'evidence-{i}'
        chart = item.get('chart', {})
        detail = 'No observed values are available for this chart in its selected scope.'
        if valid and count:
            largest = max(valid, key=lambda r: r["value"])
            grain = chart.get('timeGrain')
            time = chart.get('kind') == 'time'
            label = _period(largest['label'], grain) if time else str(largest['label'])
            metric = item['metricLabel'].lower()
            if chart.get('measure') == 'item_price' and chart.get('aggregation') == 'sum':
                metric = 'item sales (price units)'
            elif chart.get('measure') == 'order_id' and chart.get('aggregation') == 'count_distinct':
                metric = 'distinct orders'
            detail = f"{'The highest observed period is' if time else 'The largest displayed group is'} {label}, with {number(largest['value'])} {metric}. "
            same_scope = chart.get('useCurrentFilters', True)
            scope_note = '' if same_scope else ' This chart uses a broader population than the current filters; its values should not be compared directly with the filtered headline totals.'
            if time:
                detail += f"The chart covers {len(valid)} observed periods. Peaks identify when activity was concentrated; they do not identify whether promotion, demand or a change in the product mix caused it."
                insights.append(insight(f'peak-{i}', f'Peak activity in {label}', detail + scope_note, ident))
                if grain == 'month' and same_scope:
                    periods = {str(r['label'])[:7]: r for r in valid}
                    minimum = str(profile.get('dateMin', ''))[:10]
                    maximum = str(profile.get('dateMax', ''))[:10]
                    comparable = []
                    for key, row in sorted(periods.items()):
                        try:
                            year, month = map(int, key.split('-'))
                            previous = periods.get(f'{year-1:04d}-{month:02d}')
                            first = date(year, month, 1).isoformat()
                            last = date(year, month, monthrange(year, month)[1]).isoformat()
                            prev_first = date(year-1, month, 1).isoformat()
                            prev_last = date(year-1, month, monthrange(year-1, month)[1]).isoformat()
                            if previous and previous['value'] > 0 and minimum <= prev_first and maximum >= last and minimum <= first and maximum >= prev_last:
                                comparable.append((row, previous))
                        except ValueError:
                            pass
                    if comparable:
                        current, previous = comparable[-1]
                        change = (current['value']-previous['value'])/previous['value']
                        trend = f"{_period(current['label'], grain)} records {number(current['value'])} {metric}, compared with {number(previous['value'])} in {_period(previous['label'], grain)}: {'an increase' if change >= 0 else 'a decrease'} of {abs(change):.1%}. Both months are fully covered by the selected date range. This is a comparable year-on-year observation, rather than a comparison of partial years."
                        detail += ' ' + trend
                        insights.append(insight(f'comparable-{i}', 'Compare like-for-like periods', trend, ident))
            else:
                denominator = None
                if metrics and same_scope and chart.get('aggregation') == 'count_distinct' and chart.get('measure') == 'order_id' and chart.get('groupBy') == 'customer_state':
                    denominator = metrics['distinctOrders']
                if denominator:
                    coverage = sum(r['value'] for r in valid) / denominator
                    detail += f"It accounts for {largest['value']/denominator:.1%} of all selected orders. The {len(valid)} displayed states together account for {coverage:.1%}; other states are outside this chart's displayed ranking. A concentration in one state can affect service capacity and regional comparisons, so compare both order volume and sales per order before drawing a marketing conclusion."
                else:
                    detail += f"Only {len(valid)} configured groups are displayed. A high total can reflect group size, so compare the chosen measure and the underlying population before treating this group as better performing."
                insights.append(insight(f'group-{i}', f'{label} leads the displayed comparison', detail + scope_note, ident))
            if not same_scope:
                detail += scope_note
                limits.append(f"'{item['title']}' includes records outside the current filters. Its ranking and totals describe that broader population.")
        findings.append({'id': f'finding-{i}', 'title': item['title'], 'interpretation': detail, 'evidenceIds': [ident]})
        evidence.append({"title": item["title"], "definition": item["definition"] + "; " + item["scope"], "values": values})
    missing = [f for f in profile.get("fields", []) if f.get("nullCount", 0) > 0]
    if missing:
        quality = 'Missing values occur in ' + '; '.join(f"{f['name']}: {f['nullCount']:,} records ({f['nullCount']/count:.1%})" for f in missing[:5]) + '. Comparisons using these fields can represent different amounts of available data.'
        limits.append(quality)
    else:
        quality = f"None of the {len(profile.get('fields', []))} selected fields contain missing values across {count:,} records. This supports comparing the displayed totals, but it does not verify the accuracy of measurements, category definitions or source coverage."
    insights.append(insight('quality', 'Data completeness and what it tells us', quality, 'missingness'))
    findings.append({'id': 'quality', 'title': 'Completeness of the selected data', 'interpretation': quality, 'evidenceIds': ['missingness']})
    checks = ["Inspect missing values and confirm field meanings before making operational decisions."]
    if profile.get("dateMin"):
        checks.insert(0, "Compare equal-length date periods before interpreting a trend; partial periods can distort comparisons.")
    if context.get("viewId") == "sales":
        checks.insert(0, "Review the dominant customer states alongside sales per order, item mix and fulfilment capacity.")
        limits.extend(["These totals describe item prices, excluding freight. They are not profit or net revenue: discounts, returns, tax and operating costs are not reconciled in this view. Currency is not specified in the supplied metadata.", "Each record is an order item, so item counts exceed order counts. An order may contribute to several product groups; grouped order counts should not always be added together."])
        if context.get('filters', {}).get('values', {}).get('order_status') == 'delivered':
            limits.append("The scope includes delivered orders only. Canceled and undelivered orders are excluded, so the findings do not describe total demand or the rate of fulfilment failures.")
    if profile.get('dateMin') and profile.get('dateMax'):
        limits.append(f"The observed coverage is {str(profile['dateMin'])[:10]} to {str(profile['dateMax'])[:10]}. Boundary periods may be incomplete; compare equally covered periods before interpreting growth or decline.")
    limits.append("The comparisons describe observed patterns in this selected population. They cannot establish causes, measure the effect of an intervention or guarantee that a trend will continue.")
    if not count:
        checks.insert(0, "Broaden the filters or inspect the original data view, then run a new analysis.")
    return {
        "summary": summary,
        "insights": insights,
        "findings": findings,
        "limitations": limits,
        "evidence": evidence,
        "nextChecks": checks,
        "technical": {
            "method": "Descriptive SQL analysis",
            "scopeMethod": request.method,
            "snapshotId": request.snapshotId,
            "datasetVersion": profile.get("datasetVersion"),
            "schemaVersion": context.get("schemaVersion"),
            "goal": request.goal,
            "metricDefinitions": context.get("metricDefinitions", []),
            "queries": [item.get("sql") for item in request.evidence],
            "limitations": limits,
        },
    }
