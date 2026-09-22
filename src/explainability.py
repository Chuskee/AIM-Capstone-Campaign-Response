"""Explain the frozen final candidate without changing its fit or decision policy.

The validation sample is used for description only. No explanation determines
feature selection, a decision cutoff, or the identity of the chosen model.
"""

import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/capstone-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _positive_index(estimator):
    classes = np.asarray(estimator.classes_)
    positions = np.flatnonzero(classes == 1)
    if len(classes) != 2 or len(positions) != 1:
        raise ValueError("Explainability requires binary classes including Response=1.")
    return int(positions[0])


def _positive_shap(values, n_rows, n_features, positive_index):
    """Normalise SHAP's old list and current multi-output array conventions."""
    if isinstance(values, list):
        values = values[positive_index]
    elif hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[:2] == (n_rows, n_features):
        values = values[:, :, positive_index]
    if values.shape != (n_rows, n_features):
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {(n_rows, n_features)}.")
    if not np.isfinite(values).all():
        raise ValueError("SHAP produced non-finite contributions.")
    return values


def _selection_table(pipeline):
    """Include zero-variance exclusions as well as every ANOVA candidate."""
    steps = pipeline.named_steps
    original_names = np.asarray(steps["cap"].get_feature_names_out(), dtype=object)
    variance_mask = np.asarray(steps["variance"].get_support(), dtype=bool)
    remaining_names = original_names[variance_mask]
    selector = steps["select"]
    selection_mask = np.asarray(selector.get_support(), dtype=bool)
    if len(remaining_names) != len(selector.scores_):
        raise ValueError("Feature selector scores do not match the transformed schema.")
    scores = pd.Series(selector.scores_, index=remaining_names, dtype=float)
    pvalues = pd.Series(selector.pvalues_, index=remaining_names, dtype=float)
    selected_names = set(remaining_names[selection_mask])
    table = pd.DataFrame({"feature": original_names, "passed_variance_filter": variance_mask})
    table["anova_F"] = table.feature.map(scores)
    table["anova_p_value"] = table.feature.map(pvalues)
    table["anova_rank"] = table.feature.map(scores.rank(ascending=False, method="min"))
    table["selected_for_model"] = table.feature.isin(selected_names)
    table["selection_percentile"] = float(selector.percentile)
    table = table.sort_values(
        ["selected_for_model", "anova_rank", "feature"],
        ascending=[False, True, True], na_position="last",
    ).reset_index(drop=True)
    counts = {
        "engineered_features": len(original_names),
        "after_variance_filter": int(variance_mask.sum()),
        "selected_for_model": int(selection_mask.sum()),
        "selection_percentile": float(selector.percentile),
    }
    final_names = list(pipeline[:-1].get_feature_names_out())
    if set(final_names) != selected_names:
        raise ValueError("Persisted pipeline schema differs from its selected features.")
    return table, counts, final_names


def _transformed_frame(pipeline, raw, names):
    values = pipeline[:-1].transform(raw)
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if list(frame.columns) != names:
            raise ValueError("Transformed feature order differs from the fitted schema.")
    else:
        frame = pd.DataFrame(values, columns=names, index=raw.index)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite transformed features cannot be explained.")
    return frame


def _shap_explanation(clf, fit_values, valid_values, model_name, figure_path, table_path, seed):
    import shap

    positive = _positive_index(clf)
    inner_model = getattr(clf, "model_", clf)
    kind = inner_model.__class__.__name__
    is_svm = kind == "SVC"
    n_explain = min(40 if is_svm else 80, len(valid_values))
    sample = valid_values.sample(n=n_explain, random_state=seed)
    background = fit_values.sample(n=min(50, len(fit_values)), random_state=seed)
    names = list(sample.columns)
    captured = []
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        if kind in {"XGBClassifier", "RandomForestClassifier"}:
            explainer = shap.TreeExplainer(inner_model, model_output="raw")
            values = explainer.shap_values(sample, check_additivity=True)
            method = "TreeExplainer"
            output_scale = "log-odds (raw model margin)" if kind == "XGBClassifier" else "model score (calibration unverified)"
            background_rows = 0
        elif kind == "LogisticRegression":
            explainer = shap.LinearExplainer(inner_model, background)
            values = explainer.shap_values(sample)
            method = "LinearExplainer"
            output_scale = "log-odds (raw model margin)"
            background_rows = len(background)
        elif is_svm:
            def score(data):
                frame = pd.DataFrame(np.asarray(data), columns=names)
                return clf.predict_proba(frame)[:, positive]

            # KernelExplainer uses numpy's legacy RNG; restore the caller's state.
            old_state = np.random.get_state()
            np.random.seed(seed)
            try:
                explainer = shap.KernelExplainer(score, background)
                values = explainer.shap_values(sample, nsamples=120, l1_reg=0.0, silent=True)
            finally:
                np.random.set_state(old_state)
            method = "KernelExplainer (120 coalition samples per explained row)"
            output_scale = "model score (calibration unverified)"
            background_rows = len(background)
        else:
            raise TypeError(f"No configured SHAP explainer for {kind}.")
        captured.extend(f"{w.category.__name__}: {w.message}" for w in emitted)

    values = _positive_shap(values, len(sample), sample.shape[1], positive)
    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_SHAP": np.abs(values).mean(axis=0),
        "output_scale": output_scale,
        "explained_rows": len(sample),
    }).sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    importance.to_csv(table_path, index=False)
    plt.figure()
    shap.summary_plot(values, sample, max_display=min(15, len(names)), show=False, rng=np.random.default_rng(seed))
    fig = plt.gcf()
    plt.title(f"Model associations — {model_name}\nSHAP on {len(sample)} validation records", fontsize=12)
    plt.xlabel(f"SHAP contribution: {output_scale}")
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    details = {
        "status": "ok", "method": method, "output_scale": output_scale,
        "explained_rows": len(sample), "background_rows": background_rows,
        "shap_version": shap.__version__, "warnings": list(dict.fromkeys(captured)),
    }
    return details, importance


def _permutation_sensitivity(clf, values, figure_path, table_path, seed):
    """Fallback attribution without labels: score-change sensitivity, not accuracy."""
    sample = values.sample(n=min(100, len(values)), random_state=seed)
    positive = _positive_index(clf)
    baseline = clf.predict_proba(sample)[:, positive]
    rng = np.random.default_rng(seed)
    rows = []
    for name in sample.columns:
        changes = []
        for _ in range(5):
            changed = sample.copy()
            changed[name] = rng.permutation(changed[name].to_numpy())
            scores = clf.predict_proba(changed)[:, positive]
            changes.append(float(np.abs(scores - baseline).mean()))
        rows.append({"feature": name, "mean_absolute_score_change": np.mean(changes),
                     "repeat_sd": np.std(changes), "repeats": 5, "rows": len(sample)})
    table = pd.DataFrame(rows).sort_values("mean_absolute_score_change", ascending=False)
    table.to_csv(table_path, index=False)
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.feature, top.mean_absolute_score_change, color="#23759B")
    ax.set_xlabel("Mean absolute change in model score after permutation")
    ax.set_title("Fallback: prediction sensitivity to transformed features\nNot SHAP and not a measure of predictive accuracy")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return table


def _pdp_ice(pipeline, X_fit, X_valid, figure_path, table_path, seed):
    """Vary raw fields through the whole fitted pipeline; derived inputs update."""
    fields = {
        "Recency": "Recency (days since last purchase)",
        "NumWebVisitsMonth": "Website visits in the previous month",
    }
    positive = _positive_index(pipeline)
    rng = np.random.default_rng(seed)
    ice_positions = rng.choice(len(X_valid), size=min(100, len(X_valid)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    rows = []
    baseline = pipeline.predict_proba(X_valid)[:, positive].mean()
    for ax, (field, label) in zip(axes, fields.items()):
        observed = pd.to_numeric(X_fit[field], errors="raise").dropna()
        if observed.empty:
            raise ValueError(f"Cannot form a PDP grid: {field} has no fit values.")
        low, high = observed.quantile([0.01, 0.99]).to_numpy()
        grid = np.unique(np.rint(np.linspace(low, high, 21))).astype(int)
        if len(grid) < 2:
            grid = np.unique(observed.to_numpy(dtype=int))
        curves = []
        for value in grid:
            changed = X_valid.copy()
            changed[field] = value
            scores = pipeline.predict_proba(changed)[:, positive]
            curves.append(scores)
            rows.append({"raw_feature": field, "grid_value": int(value),
                         "mean_model_score": float(scores.mean()), "validation_rows": len(X_valid),
                         "unmodified_mean_score": float(baseline)})
        curves = np.asarray(curves)
        ax.plot(grid, curves[:, ice_positions], color="#23759B", alpha=0.08, linewidth=0.8)
        ax.plot(grid, curves.mean(axis=1), color="#C25523", linewidth=2.5, label="PDP: mean of validation records")
        ax.axhline(baseline, color="#666666", linestyle=":", linewidth=1, label="Unmodified mean score")
        ax.set(xlabel=label, ylim=(0, 1), title=field)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Model response score (calibration unverified)")
    fig.suptitle("Raw-input PDP and ICE — associations, not causal effects", fontsize=13)
    fig.text(0.5, 0.015, "Thin lines: up to 100 validation records. The fitted pipeline recomputes derived features.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return {"raw_features": list(fields), "validation_rows": len(X_valid),
            "ice_rows": len(ice_positions), "grid_source": "1st to 99th percentiles of fit rows, rounded to integers"}


def explain_candidate(bundle, X_fit, X_valid, output_dir, random_state=42):
    """Write explanations for a fitted bundle and return JSON-safe metadata.

    Parameters
    ----------
    bundle : dict
        Result of final candidate fitting; contains ``pipeline`` and ``metadata``.
    X_fit, X_valid : pandas.DataFrame
        Raw predictor frames from the exact final fit and policy-validation split.
        Outcomes are not required and must not be used to select explanations.
    output_dir : path-like
        Project root; outputs go under ``reports/figures`` and ``reports/tables``.

    SHAP incompatibilities are explicitly reported. A labelled permutation
    sensitivity fallback does not conceal a missing SHAP deliverable.
    """
    if not isinstance(X_fit, pd.DataFrame) or not isinstance(X_valid, pd.DataFrame):
        raise TypeError("X_fit and X_valid must be raw pandas DataFrames with named columns.")
    if len(X_fit) == 0 or len(X_valid) == 0:
        raise ValueError("Both fit and validation records are needed for explanations.")
    root = Path(output_dir).resolve()
    figures, tables = root / "reports/figures", root / "reports/tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    pipeline = bundle["pipeline"]
    name = bundle["metadata"]["model"]
    result = {
        "model": name, "paths": {}, "errors": [],
        "notes": [
            "The candidate and policy are frozen. Explanations do not select or retrain either one.",
            "ANOVA F ranks are univariate fit-data associations, not causal effects or model importance. P-values are descriptive after a model-selection process.",
            "SHAP uses the selected transformed features. Colour values are on the fitted scaled feature representation; contributions use the stated model-output scale.",
            "Correlated raw inputs and deterministic engineered features can redistribute attributions and produce implausible perturbations. Interpret ranks cautiously.",
            "PDP and ICE vary raw Recency and NumWebVisitsMonth through the full fitted pipeline, so derived features remain consistent. Other correlations may still be broken.",
            "The two PDP fields were specified in advance. A flat curve is a valid finding if the fitted model does not use that field.",
            "Class weighting and unverified calibration mean a response score is not a verified population response probability. These plots do not establish causal campaign uplift.",
        ],
    }
    selection, counts, names = _selection_table(pipeline)
    selection_path = tables / "final_feature_selection.csv"
    selection.to_csv(selection_path, index=False)
    result["paths"]["feature_selection_table"] = str(selection_path)
    result["feature_counts"] = counts
    result["feature_ranking"] = selection.to_dict(orient="records")
    fit_values = _transformed_frame(pipeline, X_fit, names)
    valid_values = _transformed_frame(pipeline, X_valid, names)
    clf = pipeline.named_steps["clf"]
    shap_figure = figures / "explain_shap.png"
    shap_table = tables / "final_shap_importance.csv"
    try:
        details, importance = _shap_explanation(
            clf, fit_values, valid_values, name, shap_figure, shap_table, random_state,
        )
        result["shap"] = details
        result["paths"].update(shap_figure=str(shap_figure), shap_importance_table=str(shap_table))
        result["shap_importance"] = importance.to_dict(orient="records")
    except Exception as exc:
        error = f"SHAP failed for {name}: {type(exc).__name__}: {exc}"
        # A failed rerun must not leave successful-looking stale SHAP artefacts.
        for path in (shap_figure, shap_table):
            if path.exists():
                path.unlink()
        result["errors"].append(error)
        result["shap"] = {"status": "failed", "error": error}
        result["notes"].append("SHAP was unavailable for this run. The fallback measures prediction changes under independent transformed-feature permutation; it does not measure accuracy or replace SHAP silently.")
        fallback_figure = figures / "explain_permutation_sensitivity.png"
        fallback_table = tables / "final_permutation_sensitivity.csv"
        try:
            fallback = _permutation_sensitivity(clf, valid_values, fallback_figure, fallback_table, random_state)
            result["paths"].update(fallback_figure=str(fallback_figure), fallback_table=str(fallback_table))
            result["fallback_importance"] = fallback.to_dict(orient="records")
        except Exception as fallback_error:
            result["errors"].append(f"Permutation fallback failed: {type(fallback_error).__name__}: {fallback_error}")
    pdp_figure, pdp_table = figures / "explain_pdp_ice.png", tables / "final_pdp.csv"
    try:
        result["pdp_ice"] = _pdp_ice(pipeline, X_fit, X_valid, pdp_figure, pdp_table, random_state)
        result["paths"].update(pdp_ice_figure=str(pdp_figure), pdp_table=str(pdp_table))
    except Exception as exc:
        for path in (pdp_figure, pdp_table):
            if path.exists():
                path.unlink()
        result["errors"].append(f"PDP/ICE failed: {type(exc).__name__}: {exc}")
        result["pdp_ice"] = {"status": "failed"}
    result = _json_safe(result)
    summary_path = root / "reports/explainability_summary.json"
    result["paths"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
