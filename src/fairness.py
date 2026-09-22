"""Validation-only contact policy fitting and transparent group fairness audits.

This module never fits a predictive model or chooses an audit axis from outcomes.
The mitigation axis is pre-specified as children living at home. A fitted policy
is a plain JSON-serialisable dictionary and must be frozen before test evaluation.
Fairness ratios are descriptive diagnostics, not legal or ethical guarantees.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _vector(values: Any, name: str, dtype: Any = None) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1 or not len(arr):
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    return arr


def _binary(values: Any, name: str) -> np.ndarray:
    arr = _vector(values, name)
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary 0/1 values.")
    return arr.astype(int)


def _scores(values: Any) -> np.ndarray:
    arr = _vector(values, "scores", float)
    if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError("scores must be finite bounded model scores between 0 and 1; calibration is not assumed.")
    return arr


def _groups(values: Any) -> np.ndarray:
    arr = _vector(values, "groups")
    return np.asarray(["Unknown" if pd.isna(v) else str(v) for v in arr])


def _same_length(*arrays: Any) -> None:
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("All input vectors must have the same length.")


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((y == 1) & (pred == 1)).sum())
    denominator = int(y.sum() + pred.sum())
    return float(2 * tp / denominator) if denominator else 0.0


def make_sensitive(df: pd.DataFrame, income_edges: Sequence[float]) -> pd.DataFrame:
    """Create descriptive audit groups from raw rows, preserving their index.

    ``income_edges`` are the three inner quartile boundaries fitted on TRAINING
    incomes, or all five training quartile boundaries (only the inner three are
    used). The same boundaries apply to validation and test, including incomes
    outside the training range. Missing income remains an explicit Unknown group.
    Age uses reference year 2014, consistent with this historical dataset.

    Columns returned: age_band, income_quartile, education, children_at_home,
    household. Children-at-home groups do not imply parental status.
    """
    required = {"Year_Birth", "Income", "Education", "Kidhome", "Teenhome", "Marital_Status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing sensitive-attribute source columns: {sorted(missing)}")
    edges = np.asarray(income_edges, dtype=float)
    if edges.ndim != 1 or len(edges) not in (3, 5):
        raise ValueError("income_edges must contain three inner or five full quartile boundaries.")
    inner = edges[1:-1] if len(edges) == 5 else edges
    if not np.isfinite(inner).all() or (np.diff(inner) < 0).any():
        raise ValueError("Inner income boundaries must be finite and nondecreasing.")

    result = pd.DataFrame(index=df.index)
    age = 2014 - pd.to_numeric(df["Year_Birth"], errors="coerce")
    bands = pd.cut(age, [-np.inf, 35, 50, 65, np.inf], right=False,
                   labels=["Under 35", "35–49", "50–64", "65+"])
    result["age_band"] = bands.astype(object).where(age.notna() & (age >= 0), "Unknown")
    income = pd.to_numeric(df["Income"], errors="coerce")
    income_group = np.asarray([f"Q{i + 1}" for i in np.searchsorted(inner, income, side="left")], dtype=object)
    income_group[~np.isfinite(income)] = "Unknown"
    result["income_quartile"] = income_group
    result["education"] = df["Education"].fillna("Unknown").astype(str)
    children = df[["Kidhome", "Teenhome"]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=2)
    result["children_at_home"] = np.where(children.isna(), "Unknown",
        np.where(children > 0, "Children at home", "No children at home"))
    marital = df["Marital_Status"]
    result["household"] = np.where(marital.isin(["Married", "Together"]), "Partnered",
        np.where(marital.isin(["Single", "Divorced", "Widow", "Alone"]), "Not partnered", "Unknown"))
    return result


def _selection_summary(pred: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    rates = {str(g): float(pred[groups == g].mean()) for g in np.unique(groups)}
    values = list(rates.values())
    ratio = float(min(values) / max(values)) if len(values) > 1 and max(values) > 0 else None
    gap = float(max(values) - min(values)) if len(values) > 1 else None
    return {"group_selection_rates": rates, "selection_rate_ratio": ratio,
            "demographic_parity_gap": gap}


def fit_policy(y_valid: Any, scores_valid: Any, children_groups_valid: Any,
               parity_floor: float = 0.8) -> dict[str, Any]:
    """Fit a global F1 threshold and an optional group policy on VALIDATION only.

    The global threshold maximises validation F1 over distinct score thresholds;
    tied maxima use the higher threshold. Group thresholds are validation score
    quantiles targeting the overall selection rate of that global policy. The
    fixed alpha grid 0, .05, ..., 1 interpolates global and group thresholds. The
    smallest alpha attaining a minimum/maximum selection ratio >= parity_floor
    is retained. If no candidate works (including all-zero selections or only
    one observed group), retain the global policy and explicitly flag failure.

    The 0.80 floor is an illustrative design assumption, not a fairness guarantee.
    Quantile ties and small groups can prevent the target from being attained.
    Selection counts need not remain identical after threshold interpolation.
    """
    y, scores, groups = _binary(y_valid, "y_valid"), _scores(scores_valid), _groups(children_groups_valid)
    _same_length(y, scores, groups)
    if not 0 < parity_floor <= 1:
        raise ValueError("parity_floor must be greater than 0 and at most 1.")
    thresholds = np.append(np.unique(scores), np.nextafter(1.0, np.inf))
    global_threshold = float(max(thresholds, key=lambda t: (_f1(y, scores >= t), float(t))))
    baseline = (scores >= global_threshold).astype(int)
    target_rate = float(baseline.mean())
    quantile_thresholds = {}
    for group in np.unique(groups):
        if target_rate == 0:
            threshold = np.nextafter(1.0, np.inf)
        elif target_rate == 1:
            threshold = 0.0
        else:
            threshold = np.quantile(scores[groups == group], 1 - target_rate)
        quantile_thresholds[str(group)] = float(threshold)
    candidates = []
    chosen = None
    for alpha in np.linspace(0, 1, 21):
        group_thresholds = {g: float((1 - alpha) * global_threshold + alpha * q)
                            for g, q in quantile_thresholds.items()}
        pred = np.asarray([score >= group_thresholds[g] for score, g in zip(scores, groups)], dtype=int)
        summary = _selection_summary(pred, groups)
        ratio = summary["selection_rate_ratio"]
        row = {"alpha": float(round(alpha, 2)), "f1": _f1(y, pred),
               "contact_fraction": float(pred.mean()), **summary}
        candidates.append(row)
        if chosen is None and ratio is not None and ratio + 1e-12 >= parity_floor:
            chosen = (row, group_thresholds)
    if chosen is None:
        selected = candidates[0]
        final_thresholds = {g: global_threshold for g in quantile_thresholds}
        status = "insufficient_groups" if len(quantile_thresholds) < 2 else "target_not_achieved_global_fallback"
    else:
        selected, final_thresholds = chosen
        status = "target_achieved_on_validation"
    return {
        "version": 1, "fit_partition": "validation", "mitigation_axis": "children_at_home",
        "objective": "global F1 followed by the smallest alpha attaining the pre-specified selection-rate ratio",
        "global_threshold": global_threshold, "global_validation_f1": _f1(y, baseline),
        "target_selection_rate": target_rate, "target_selection_rate_definition": "overall global-policy validation contact fraction",
        "parity_floor": float(parity_floor), "quantile_thresholds": quantile_thresholds,
        "alpha": selected["alpha"], "group_thresholds": final_thresholds,
        "validation_metrics": selected, "validation_grid": candidates, "status": status,
        "unseen_group_handling": "error; manual review or prospectively validated fallback required",
        "limitation": "Selection parity does not establish equalised odds, causal fairness, or regulatory compliance.",
    }


def apply_policy(scores: Any, groups: Any, policy: dict[str, Any]) -> np.ndarray:
    """Apply saved group thresholds without fitting; reject unseen groups explicitly."""
    scores, groups = _scores(scores), _groups(groups)
    _same_length(scores, groups)
    thresholds = policy["group_thresholds"]
    unseen = set(np.unique(groups)).difference(thresholds)
    if unseen:
        raise ValueError(f"Policy received unseen group(s): {sorted(unseen)}. Do not silently refit using test data.")
    return np.asarray([s >= thresholds[g] for s, g in zip(scores, groups)], dtype=int)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Return a 95% Wilson binomial interval, or (None, None) for no denominator."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Counts must satisfy 0 <= successes <= total.")
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return float(max(0, centre - half)), float(min(1, centre + half))


def evaluate_predictions(y_true: Any, scores: Any, predictions: Any,
                         value_per_response: float = 11.0, cost_per_contact: float = 3.0) -> dict[str, Any]:
    """Evaluate a frozen contact policy and an explicitly hypothetical profit proxy.

    Profit = value_per_response * true positives - cost_per_contact * contacts.
    Defaults (11 and 3) are assumptions in arbitrary monetary units, not measured
    incremental ROI: historical response labels do not establish causal uplift.
    Undefined precision/recall and one-class ranking metrics are returned as None.
    """
    y, scores, pred = _binary(y_true, "y_true"), _scores(scores), _binary(predictions, "predictions")
    _same_length(y, scores, pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    spend = float(cost_per_contact * int(pred.sum()))
    profit = float(value_per_response * tp - spend)
    two_classes = len(np.unique(y)) == 2
    return {"n": int(len(y)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "selected_n": int(pred.sum()), "positive_n": int(y.sum()),
            "accuracy": float((pred == y).mean()), "precision": _divide(tp, tp + fp),
            "recall": _divide(tp, tp + fn), "f1": _f1(y, pred),
            "specificity": _divide(tn, tn + fp), "false_positive_rate": _divide(fp, fp + tn),
            "roc_auc": float(roc_auc_score(y, scores)) if two_classes else None,
            "average_precision": float(average_precision_score(y, scores)) if two_classes else None,
            "contact_fraction": float(pred.mean()), "response_prevalence": float(y.mean()),
            "assumed_value_per_response": float(value_per_response), "assumed_cost_per_contact": float(cost_per_contact),
            "observed_label_profit_proxy": profit, "assumed_contact_spend": spend,
            "observed_label_roi_proxy": _divide(profit, spend),
            "profit_caveat": "Hypothetical units and historical response labels; not measured causal uplift or realised ROI."}


def audit_fairness(y_true: Any, predictions: Any, sensitive: pd.DataFrame) -> dict[str, Any]:
    """Audit every supplied sensitive axis for the exact supplied frozen policy.

    Group rows include sample sizes, confusion counts, selection/TPR Wilson CIs,
    FPR and precision. Summary metrics are demographic-parity gap, min/max
    selection ratio (descriptive disparate-impact ratio), and equalised-odds gap.
    Equalised-odds gap is max(TPR gap, FPR gap), and is None if any included group
    lacks positives or negatives. Undefined comparisons are never replaced by 0.
    Wilson intervals describe group rates, not uncertainty of between-group gaps.
    """
    y, pred = _binary(y_true, "y_true"), _binary(predictions, "predictions")
    _same_length(y, pred, sensitive)
    if not isinstance(sensitive, pd.DataFrame) or sensitive.shape[1] == 0:
        raise ValueError("sensitive must be a DataFrame with at least one audit axis.")
    rows, summaries = [], []
    for attribute in sensitive.columns:
        groups = _groups(sensitive[attribute])
        attribute_rows = []
        for group in np.unique(groups):
            mask = groups == group
            yg, pg = y[mask], pred[mask]
            n, positives, negatives = int(mask.sum()), int(yg.sum()), int((yg == 0).sum())
            selected = int(pg.sum())
            tp = int(((yg == 1) & (pg == 1)).sum())
            fp = int(((yg == 0) & (pg == 1)).sum())
            selection_ci, tpr_ci = wilson_interval(selected, n), wilson_interval(tp, positives)
            row = {"attribute": str(attribute), "group": str(group), "n": n,
                   "positive_n": positives, "negative_n": negatives, "selected_n": selected,
                   "tp": tp, "fp": fp, "tn": negatives - fp, "fn": positives - tp,
                   "selection_rate": _divide(selected, n), "tpr": _divide(tp, positives),
                   "fpr": _divide(fp, negatives), "precision": _divide(tp, selected),
                   "selection_wilson_low": selection_ci[0], "selection_wilson_high": selection_ci[1],
                   "tpr_wilson_low": tpr_ci[0], "tpr_wilson_high": tpr_ci[1]}
            rows.append(row)
            attribute_rows.append(row)
        summary = _selection_summary(pred, groups)
        comparable = len(attribute_rows) >= 2
        def gap(key: str) -> float | None:
            values = [r[key] for r in attribute_rows]
            return float(max(values) - min(values)) if comparable and all(v is not None for v in values) else None
        tpr_gap, fpr_gap = gap("tpr"), gap("fpr")
        summaries.append({"attribute": str(attribute), "group_count": len(attribute_rows),
                          "minimum_group_n": min(r["n"] for r in attribute_rows),
                          "demographic_parity_gap": summary["demographic_parity_gap"],
                          "disparate_impact_ratio": summary["selection_rate_ratio"],
                          "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
                          "equalised_odds_gap": max(tpr_gap, fpr_gap) if tpr_gap is not None and fpr_gap is not None else None,
                          "all_groups_have_both_outcomes": all(r["positive_n"] > 0 and r["negative_n"] > 0 for r in attribute_rows)})
    return {"groups": rows, "summary": summaries,
            "caveat": "Descriptive group audit. Small samples, multiple comparisons and absent attributes limit conclusions; no fairness guarantee."}
