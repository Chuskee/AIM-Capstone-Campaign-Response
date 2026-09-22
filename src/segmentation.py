"""Descriptive, label-free customer segmentation and PCA diagnostics.

Run from the project root, importing this module as ``src.segmentation`` so that
the saved pipeline can resolve its custom transformer when loaded with joblib.
The full customer snapshot is used for exploratory segmentation. Internal
clustering scores and seed agreement are not held-out predictive performance.
"""

from itertools import combinations
from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


REFERENCE_DATE = "2014-06-30"
SPEND_COLUMNS = (
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds",
)
PURCHASE_COLUMNS = ("NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases")
FEATURES = (
    "Income", "Age_2014", "Recency", "Tenure_Days", "Total_Spend",
    "Total_Purchases", "Children_At_Home", "NumDealsPurchases",
    "NumWebVisitsMonth", "Wine_Spend_Share", "Meat_Spend_Share",
    "Web_Purchase_Share", "Catalog_Purchase_Share",
)
LOG_FEATURES = (
    "Income", "Tenure_Days", "Total_Spend", "Total_Purchases",
    "NumDealsPurchases", "NumWebVisitsMonth",
)


class SegmentFeatures(TransformerMixin, BaseEstimator):
    """Derive customer features without observing campaign-response labels.

    Monetary values retain the source's unspecified currency. Spending shares
    and channel shares only combine variables from their respective families;
    no spending/purchase ratio with potentially mismatched windows is created.
    """

    def __init__(self, reference_date=REFERENCE_DATE):
        self.reference_date = reference_date

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        frame = X.copy()
        dates = pd.to_datetime(frame["Dt_Customer"], dayfirst=True, errors="raise")
        reference = pd.Timestamp(self.reference_date)
        total_spend = frame[list(SPEND_COLUMNS)].sum(axis=1, min_count=1)
        total_purchases = frame[list(PURCHASE_COLUMNS)].sum(axis=1, min_count=1)
        out = pd.DataFrame(index=frame.index)
        out["Income"] = pd.to_numeric(frame["Income"], errors="raise")
        out["Age_2014"] = reference.year - frame["Year_Birth"]
        out["Recency"] = frame["Recency"]
        out["Tenure_Days"] = (reference - dates).dt.days
        out["Total_Spend"] = total_spend
        out["Total_Purchases"] = total_purchases
        out["Children_At_Home"] = frame["Kidhome"] + frame["Teenhome"]
        out["NumDealsPurchases"] = frame["NumDealsPurchases"]
        out["NumWebVisitsMonth"] = frame["NumWebVisitsMonth"]
        for name, numerator, denominator in (
            ("Wine_Spend_Share", frame["MntWines"], total_spend),
            ("Meat_Spend_Share", frame["MntMeatProducts"], total_spend),
            ("Web_Purchase_Share", frame["NumWebPurchases"], total_purchases),
            ("Catalog_Purchase_Share", frame["NumCatalogPurchases"], total_purchases),
        ):
            out[name] = numerator.div(denominator.replace(0, np.nan))
            out.loc[denominator.eq(0), name] = 0.0
        return out[list(FEATURES)].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def log_selected_features(X, indices):
    """Log1p selected nonnegative, skewed columns; retain the other columns."""
    result = np.asarray(X, dtype=float).copy()
    if np.any(result[:, indices] < 0):
        raise ValueError("Segmentation log features must be nonnegative after cleaning.")
    result[:, indices] = np.log1p(result[:, indices])
    return result


def make_segment_preprocessor():
    return Pipeline([
        ("features", SegmentFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(
            log_selected_features,
            kw_args={"indices": [FEATURES.index(name) for name in LOG_FEATURES]},
            feature_names_out="one-to-one",
            check_inverse=False,
        )),
        ("scale", StandardScaler()),
    ])


def _scores(X, labels, random_state):
    return {
        "silhouette": float(silhouette_score(
            X, labels, sample_size=min(1500, len(X)), random_state=random_state,
        )),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "smallest_cluster_n": int(np.bincount(labels).min()),
        "largest_cluster_n": int(np.bincount(labels).max()),
    }


def _profiles(df, feature_frame, labels):
    grouped = feature_frame.assign(Cluster=labels).groupby("Cluster")
    profile = grouped.mean().add_suffix("_mean")
    profile.insert(0, "Customer_Count", grouped.size())
    profile.insert(1, "Customer_Share", grouped.size() / len(df))
    for column in ("Income", "Total_Spend", "Total_Purchases"):
        profile[column + "_median"] = grouped[column].median()
    if "Response" in df:
        # The outcome is joined only after clusters have been fitted and chosen.
        profile["Observed_Response_Rate"] = pd.Series(
            df["Response"].to_numpy(), index=feature_frame.index,
        ).groupby(pd.Series(labels, index=feature_frame.index)).mean()
    return profile.reset_index()


def run_segmentation(df, output_dir, random_state=42):
    """Fit descriptive segmentations; save reproducible artifacts and figures.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned source records, retaining base customer columns. Income can be
        missing. Response is optional and is never an input to clustering/PCA.
    output_dir : path-like
        Revised project root; artifacts go in models/ and reports/.

    Returns
    -------
    dict
        JSON-safe results and project-relative artifact paths.
    """
    root = Path(output_dir)
    for subdir in ("models", "reports/tables", "reports/figures"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    if len(df) < 10:
        raise ValueError("At least 10 customer records are required for this comparison.")

    df = df.copy()
    preprocessor = make_segment_preprocessor()
    X = preprocessor.fit_transform(df)
    feature_frame = preprocessor.named_steps["features"].transform(df)
    candidates = []
    kmeans_models = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X)
        candidates.append({
            "algorithm": "KMeans", "k": k, "inertia": float(model.inertia_),
            **_scores(X, labels, random_state),
        })
        kmeans_models[k] = model
    for k in range(2, 5):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        candidates.append({
            "algorithm": "Ward", "k": k, "inertia": None,
            **_scores(X, labels, random_state),
        })
    evaluation = pd.DataFrame(candidates)
    km_results = evaluation[evaluation.algorithm.eq("KMeans")]
    chosen = km_results.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    selected_k = int(chosen.k)
    selected_model = kmeans_models[selected_k]
    labels = selected_model.labels_
    three_labels = kmeans_models[3].labels_
    # Keep the entire preprocessing chain with the clusterer for future scoring.
    pipeline = Pipeline(preprocessor.steps + [("cluster", selected_model)])
    pipeline_path = root / "models/customer_segmentation.joblib"
    joblib.dump(pipeline, pipeline_path)
    np.testing.assert_array_equal(pipeline.predict(df), labels)
    loaded = joblib.load(pipeline_path)
    np.testing.assert_array_equal(loaded.predict(df), labels)

    stability = []
    for k in sorted({2, 3, selected_k}):
        seeds = [int(random_state + offset) for offset in range(5)]
        runs = [
            KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X)
            for seed in seeds
        ]
        agreements = [
            float(adjusted_rand_score(runs[a], runs[b]))
            for a, b in combinations(range(len(runs)), 2)
        ]
        stability.append({
            "k": k, "seeds": seeds, "n_pairwise_comparisons": len(agreements),
            "mean_adjusted_rand_index": float(np.mean(agreements)),
            "min_adjusted_rand_index": float(np.min(agreements)),
            "scope": "Initialization stability on the same customer snapshot; not resampling stability.",
        })

    assignments = pd.DataFrame({
        "Source_Row_Index": df.index.to_numpy(),
        "Selected_Cluster": labels,
        "Alternative_K3_Cluster": three_labels,
    })
    if "ID" in df:
        assignments.insert(1, "ID", df["ID"].to_numpy())
    profiles = _profiles(df, feature_frame, labels)
    k3_profiles = _profiles(df, feature_frame, three_labels)

    pca = PCA(random_state=random_state).fit(X)
    coordinates = pca.transform(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumulative, 0.90) + 1)
    pca_variance = pd.DataFrame({
        "Component": np.arange(1, len(FEATURES) + 1),
        "Explained_Variance_Ratio": pca.explained_variance_ratio_,
        "Cumulative_Explained_Variance": cumulative,
    })
    loadings = pd.DataFrame(
        pca.components_.T, index=FEATURES,
        columns=[f"PC{i}" for i in range(1, len(FEATURES) + 1)],
    ).rename_axis("Feature").reset_index()
    pca_coordinates = assignments.copy()
    pca_coordinates["PC1"] = coordinates[:, 0]
    pca_coordinates["PC2"] = coordinates[:, 1]
    joblib.dump(
        Pipeline(preprocessor.steps + [("pca", pca)]),
        root / "models/customer_segmentation_pca.joblib",
    )

    files = {
        "pipeline": "models/customer_segmentation.joblib",
        "pca_pipeline": "models/customer_segmentation_pca.joblib",
        "evaluation": "reports/tables/segmentation_comparison.csv",
        "assignments": "reports/tables/customer_segment_assignments.csv",
        "profiles": "reports/tables/segment_profiles.csv",
        "k3_profiles": "reports/tables/segment_profiles_k3_alternative.csv",
        "stability": "reports/tables/segmentation_seed_stability.json",
        "pca_variance": "reports/tables/segmentation_pca_variance.csv",
        "pca_loadings": "reports/tables/segmentation_pca_loadings.csv",
        "pca_coordinates": "reports/tables/segmentation_pca_coordinates.csv",
        "comparison_figure": "reports/figures/segmentation_comparison.png",
        "pca_figure": "reports/figures/segmentation_pca.png",
        "profiles_figure": "reports/figures/segment_profiles.png",
    }
    for name, frame in (
        ("evaluation", evaluation), ("assignments", assignments),
        ("profiles", profiles), ("k3_profiles", k3_profiles),
        ("pca_variance", pca_variance), ("pca_loadings", loadings),
        ("pca_coordinates", pca_coordinates),
    ):
        frame.to_csv(root / files[name], index=False)
    (root / files["stability"]).write_text(json.dumps(stability, indent=2))

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
        for algorithm, rows in evaluation.groupby("algorithm"):
            for ax, metric, title in (
                (axes[0], "silhouette", "Silhouette (higher is better)"),
                (axes[1], "davies_bouldin", "Davies–Bouldin (lower is better)"),
                (axes[2], "calinski_harabasz", "Calinski–Harabasz (higher is better)"),
            ):
                ax.plot(rows.k, rows[metric], marker="o", label=algorithm)
                ax.set(title=title, xlabel="Number of clusters", xticks=range(2, 7))
                ax.grid(alpha=0.2)
        axes[0].legend(frameon=False)
        fig.suptitle("Internal clustering comparison on the customer snapshot")
        fig.savefig(root / files["comparison_figure"], dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        for cluster in range(selected_k):
            mask = labels == cluster
            axes[0].scatter(
                coordinates[mask, 0], coordinates[mask, 1], s=12, alpha=0.55,
                label=f"Cluster {cluster} (n={int(mask.sum())})",
            )
        axes[0].set(
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            title=f"Selected K-Means segmentation: k={selected_k}",
        )
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].plot(pca_variance.Component, cumulative, marker="o", color="#126E82")
        axes[1].axhline(0.90, color="#A44A3F", linestyle="--", label="90% variance")
        axes[1].axvline(n90, color="#6B7280", linestyle=":", label=f"{n90} components")
        axes[1].set(
            xlabel="Number of principal components", ylabel="Cumulative explained variance",
            title="PCA is a descriptive projection", ylim=(0, 1.03),
        )
        axes[1].legend(frameon=False)
        fig.savefig(root / files["pca_figure"], dpi=160, bbox_inches="tight")
        plt.close(fig)

        means = pd.DataFrame(X, columns=FEATURES).assign(Cluster=labels).groupby("Cluster").mean()
        fig, ax = plt.subplots(figsize=(12, max(3.5, selected_k * 0.65)), layout="constrained")
        limit = max(1.0, float(np.max(np.abs(means.to_numpy()))))
        im = ax.imshow(means, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
        ax.set_yticks(range(selected_k), [f"Cluster {i}" for i in means.index])
        ax.set_title("Segment profiles: means of transformed and standardized features")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Mean z-score")
        fig.savefig(root / files["profiles_figure"], dpi=160, bbox_inches="tight")
        plt.close(fig)

    best_overall = evaluation.sort_values("silhouette", ascending=False).iloc[0]
    summary = {
        "task_type": "Descriptive unsupervised customer segmentation",
        "n_customers": len(df),
        "selected_k": selected_k,
        "selected_silhouette": float(chosen.silhouette),
        "selection_rule": "Highest silhouette among KMeans k=2..6; ties prefer smaller k. Ward is a diagnostic comparator.",
        "best_candidate_overall": {
            "algorithm": str(best_overall.algorithm), "k": int(best_overall.k),
            "silhouette": float(best_overall.silhouette),
        },
        "k3_silhouette": float(km_results.loc[km_results.k.eq(3), "silhouette"].iloc[0]),
        "features": list(FEATURES),
        "feature_notes": {
            "reference_date": REFERENCE_DATE,
            "reference_date_reason": "Fixed near the last source enrollment date; avoids aging this historical snapshot to the execution year.",
            "age": "Approximate age: 2014 minus year of birth; exact birthday unavailable.",
            "currency": "Not specified by the dataset; no currency symbol is assumed.",
            "log1p_features": list(LOG_FEATURES),
            "shares": "Within-family spending and purchase-channel shares; zero total means share zero.",
            "income_missing": "Median-imputed only in the fitted segmentation pipeline; profile income statistics use observed values.",
            "target_exclusion": "Response and AcceptedCmp variables are excluded from clustering and PCA.",
            "demographics": "Age, income and children at home influence distances; deployment would require checks on segment-specific offers and outcomes.",
        },
        "limitations": [
            "Scores describe structure within this historical snapshot, not generalization to future customers.",
            "Silhouette is calculated on a fixed random sample of at most 1500 records; other scores use all records.",
            "Seed stability tests initialization only, not population or time stability.",
            "KMeans is chosen for assignable operational segments; higher Ward silhouette, if present, must be disclosed.",
            "Three clusters remain an exploratory alternative; extra marketing value requires a campaign experiment.",
            "Spending and channel shares can overlap in information with totals; segment sensitivity to features remains a deployment check.",
        ],
        "stability": stability,
        "pca": {
            "first_two_explained_variance": float(cumulative[1]),
            "n_components_90": n90,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "role": "Descriptive visualization and dimensionality diagnostic; neither clustering nor the classifier is fitted on these two PCs.",
            "loadings_definition": "PCA component coefficients (unit-length directions), not feature-outcome associations.",
        },
        "profiles": json.loads(profiles.to_json(orient="records")),
        "files": files,
    }
    (root / "reports/tables/segmentation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
