"""Describe the actual fitted groups using original-unit distributions."""
import numpy as np
import pandas as pd

from app.services.report_insights import data_limits, insight, number


def cluster_report(request, df, prep, score, model, visuals, visual):
    labels = model.labels_
    k = model.n_clusters
    numeric = [c for c in prep.columns_ if pd.api.types.is_numeric_dtype(df[c])][:8]
    sizes = [{"group": f"Group {i+1}", "count": int(np.sum(labels == i))} for i in range(k)]
    profiles, contrasts, stats = [], [], {}
    for c in numeric:
        overall, std = float(df[c].mean()), float(df[c].std())
        stats[c] = []
        for i in range(k):
            raw = df.loc[labels == i, c].dropna()
            if raw.empty:
                continue
            mean = float(raw.mean())
            row = {"group": f"Group {i+1}", "count": len(raw), "mean": mean,
                   "median": float(raw.median()), "q1": float(raw.quantile(.25)), "q3": float(raw.quantile(.75))}
            stats[c].append(row)
            if std > 0:
                profiles.append({"field": c, "group": row['group'], "mean": mean, "difference": (mean-overall)/std})
        if std > 0 and len(stats[c]) >= 2:
            low, high = min(stats[c], key=lambda r: r['mean']), max(stats[c], key=lambda r: r['mean'])
            contrasts.append(((high['mean']-low['mean'])/std, c, low, high))
    if profiles:
        visuals.append(visual("cluster-profiles", "Group characteristics relative to the whole dataset", profiles, "rect", {
            "x": {"field": "field", "type": "nominal", "title": "Characteristic"},
            "y": {"field": "group", "type": "nominal", "title": None},
            "color": {"field": "difference", "type": "quantitative", "scale": {"scheme": "redblue", "domainMid": 0}, "title": "Relative difference"},
            "tooltip": [{"field": "field"}, {"field": "group"}, {"field": "mean", "type": "quantitative", "format": ",.2f"}, {"field": "difference", "type": "quantitative", "format": ".2f"}],
        }, "Blue marks a group mean above the overall mean; red marks a mean below it. Color expresses a difference in standard deviations, while the tooltip gives the original-unit mean. Missing measurements are excluded from these summaries."))
    sizes_text = "; ".join(f"{r['group']}: {r['count']:,} records ({r['count']/len(df):.1%})" for r in sizes)
    insights = [insight("population-groups", "How the population is divided", sizes_text + f". These groups summarize {len(df):,} selected records; each record represents {request.rowGrain.lower()}. Group size describes coverage, not value or prediction accuracy.", "cluster-sizes")]
    contrast_text = " ".join(f"{c}: {high['group']} averages {number(high['mean'])}, compared with {number(low['mean'])} in {low['group']} (difference {number(high['mean']-low['mean'])})." for _, c, low, high in sorted(contrasts, key=lambda r: r[0], reverse=True)[:3])
    if contrasts:
        insights.append(insight("separating-characteristics", "The clearest differences between groups", contrast_text + " These are observed group characteristics; they explain how the fitted groups differ, rather than what caused that difference.", "cluster-profiles"))
    insights.append(insight("separation", "How distinct are these groups?", f"The silhouette score is {score:.3f}. " + ("The average separation is weak, so many records lie between plausible groups. Use these profiles as an exploration of differences, and check stability before treating them as distinct customer segments." if score < .25 else "The fitted groups show measurable separation in the selected characteristics. Check distribution overlap and stability in another sample before assigning permanent segment names."), "projection", "cluster-profiles"))
    findings = [
        {"id": "groups", "title": "Group coverage in the selected population", "interpretation": sizes_text + ". Compare the number of records with their observed characteristics before deciding which group deserves attention. A larger group is not necessarily a higher-value group.", "evidenceIds": ["cluster-sizes"]},
        {"id": "projection", "title": "Read the projection together with the group profiles", "interpretation": "Nearby points have similar positions in this two-dimensional view. The overlap shows why a segment boundary should not be treated as a firm distinction for every record. The caption states how much information the projection retains; use the original-unit distributions below to understand the groups rather than assigning a business meaning to either axis.", "evidenceIds": ["projection"] if any(v['id'] == 'projection' for v in visuals) else ["cluster-sizes"]},
    ]
    if profiles:
        findings.append({"id": "profiles", "title": "Which characteristics distinguish the groups?", "interpretation": contrast_text + " Compare colors within each characteristic. A large contrast for one field can drive separation even when other fields overlap; it does not prove that this field causes customer behavior.", "evidenceIds": ["cluster-profiles"]})
    for index, c in enumerate(numeric[:5]):
        if not stats[c]:
            continue
        ident = f"distribution-{index}"
        # Store every observed value, not a preview or standardized surrogate.
        values = [{"group": f"Group {int(labels[i])+1}", "value": float(v)} for i, v in enumerate(df[c]) if pd.notna(v)]
        visuals.append(visual(ident, f"{c} distribution by group", values, {"type": "boxplot", "extent": 1.5, "size": 40}, {
            "x": {"field": "group", "type": "nominal", "title": None, "sort": [f"Group {i+1}" for i in range(k)]},
            "y": {"field": "value", "type": "quantitative", "title": c, "scale": {"zero": False}},
            "color": {"field": "group", "type": "nominal", "legend": None},
        }, f"All {len(values):,} non-missing measurements in their original units. The line is the median; the box covers the middle 50%; whiskers extend to observations within 1.5 box widths. Points outside the whiskers are unusual values, not automatically invalid records."))
        description = " ".join(f"{r['group']}: median {number(r['median'])}, with the middle 50% from {number(r['q1'])} to {number(r['q3'])} ({r['count']:,} observed records)." for r in stats[c])
        common_low = max(r['q1'] for r in stats[c])
        common_high = min(r['q3'] for r in stats[c])
        overlap = (f" The middle ranges share values between {number(common_low)} and {number(common_high)}; this characteristic alone does not clearly distinguish all groups." if len(stats[c]) > 1 and common_low <= common_high else " The central ranges differ between at least some groups. Check their spread and outliers, not only the average, when interpreting the contrast.")
        findings.append({"id": ident, "title": f"What the spread of {c} tells us", "interpretation": description + overlap, "evidenceIds": [ident]})
        if "spending" in c.lower():
            insights.append(insight("spending-spread", "Spending patterns within the groups", description + overlap, ident))
    limits = data_limits(df, prep.columns_)
    limits.insert(0, f"A silhouette of {score:.3f} summarizes separation in these selected characteristics. Group labels have not been checked for stability across new samples or changes in the population.")
    if not any(f.get('role') == 'date' for f in request.fields):
        limits.append("There is no time field in this view. The profiles cannot show how individuals change over time or distinguish stable patterns from a one-time snapshot.")
    if any('spending score' in c.lower() for c in df):
        limits.append("Spending Score is a supplied rating, not a transaction total. Without purchase history, order frequency or profit, these groups cannot establish customer lifetime value or revenue contribution.")
    limits.append("Group assignment depends on the selected characteristics and their scale. Validate the profiles against the analysis goal and a different sample before naming or acting on the segments.")
    return {
        "summary": [f"Identified {k} groups across {len(df):,} records, with {len(prep.columns_)} selected characteristics.", contrast_text or sizes_text],
        "insights": insights,
        "findings": findings,
        "visuals": visuals,
        "metrics": [{"label": "Groups", "value": k}, {"label": "Silhouette", "value": score}],
        "nextSteps": ["Compare the original-unit boxplots and identify which differences are useful for the stated goal.", "Check records with unusual measurements and incomplete characteristics before naming segments.", "Repeat the analysis on another population or period and compare group sizes and profiles."],
        "limitations": limits,
        "evaluation": {"features": prep.columns_, "clusters": k, "method": "K-means", "seed": 42, "clusterChoice": "Specified group count" if request.clusters else "Automatic selection"},
    }
