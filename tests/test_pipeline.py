"""Fast invariant checks for grouped splits and saved classifier pipelines."""

import copy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from src.fairness import make_sensitive
from src.pipeline import (CAMPAIGNS, PURCHASES, SPEND, CustomerFeatures,
                          clean_data, group_split, income_edges,
                          model_candidates, predict_bundle)


def raw_customers(n=60):
    """Small reproducible sample with real column names and nonconstant features."""
    rng = np.random.default_rng(617)
    frame = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "Year_Birth": rng.integers(1940, 1995, n),
        "Income": rng.integers(18000, 100000, n).astype(float),
        "Education": rng.choice(["Basic", "2n Cycle", "Graduation", "Master", "PhD"], n),
        "Marital_Status": rng.choice(["Married", "Together", "Single", "Divorced", "Widow"], n),
        "Kidhome": rng.integers(0, 2, n),
        "Teenhome": rng.integers(0, 2, n),
        "Dt_Customer": (pd.Timestamp("2012-08-01") + pd.to_timedelta(np.arange(n), unit="D")).strftime("%d-%m-%Y"),
        "Recency": rng.integers(0, 100, n),
        "Complain": rng.integers(0, 2, n),
        "NumDealsPurchases": rng.integers(0, 7, n),
        "NumWebVisitsMonth": rng.integers(0, 16, n),
        "Response": (np.arange(n) % 5 == 0).astype(int),
        "Z_CostContact": 3,
        "Z_Revenue": 11,
    })
    for col in SPEND:
        frame[col] = rng.integers(0, 800, n)
    for col in PURCHASES:
        frame[col] = rng.integers(1, 12, n)
    for col in CAMPAIGNS:
        frame[col] = rng.integers(0, 2, n)
    frame.loc[[0, 7], "Income"] = np.nan
    return frame


class PipelineInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.groups, _ = clean_data(raw_customers())
        cls.X = cls.data.drop(columns="Response")
        cls.y = cls.data.Response
        cls.model = model_candidates()["Logistic Regression"][0]
        cls.model.fit(cls.X, cls.y)
        edges = income_edges(cls.X)
        labels = make_sensitive(cls.X, edges).children_at_home.unique()
        cls.bundle = {"pipeline": cls.model, "income_edges": edges,
                      "policy": {"group_thresholds": {str(label): 0.5 for label in labels}}}

    def test_identical_profiles_with_conflicting_targets_never_cross_split(self):
        original = raw_customers(40)
        conflicts = original.iloc[:8].copy()
        conflicts["ID"] += 1000
        conflicts["Response"] = 1 - conflicts.Response
        combined, groups, _ = clean_data(pd.concat([original, conflicts], ignore_index=True))
        self.assertEqual(len(combined), 48)
        np.testing.assert_array_equal(groups[:8], groups[40:48])
        self.assertTrue((combined.Response.iloc[:8].to_numpy() != combined.Response.iloc[40:].to_numpy()).all())
        for seed in range(5):
            fit, holdout = group_split(combined.drop(columns="Response"), combined.Response,
                                       groups, n_splits=5, seed=seed)
            self.assertTrue(set(groups[fit]).isdisjoint(groups[holdout]))
            holdout_rows = set(holdout)
            for row in range(8):
                self.assertEqual(row in holdout_rows, row + 40 in holdout_rows)

    def test_transforming_extreme_holdout_does_not_update_imputer_or_caps(self):
        imputer = self.model.named_steps["imputer"]
        cap = self.model.named_steps["cap"]
        median_before = imputer.statistics_.copy()
        caps_before = copy.deepcopy(cap.caps_)
        income_position = list(imputer.feature_names_in_).index("Income")
        self.assertAlmostEqual(median_before[income_position], self.X.Income.median())
        training_features = self.model.named_steps["features"].transform(self.X)
        training_imputed = imputer.transform(training_features)
        self.assertAlmostEqual(caps_before["Income"], training_imputed.Income.quantile(0.99))

        holdout = self.X.iloc[:2].copy()
        holdout.loc[holdout.index[0], "Income"] = 1e12
        holdout.loc[holdout.index[1], "Income"] = np.nan
        self.model.predict_proba(holdout)
        holdout_imputed = imputer.transform(self.model.named_steps["features"].transform(holdout))
        holdout_capped = cap.transform(holdout_imputed)
        self.assertEqual(holdout_capped.Income.iloc[0], caps_before["Income"])
        self.assertEqual(holdout_imputed.Income.iloc[1], self.X.Income.median())
        np.testing.assert_array_equal(imputer.statistics_, median_before)
        self.assertEqual(cap.caps_, caps_before)

    def test_features_ignore_identifier_target_and_keep_missing_income_flag(self):
        raw = self.data.copy()
        features = CustomerFeatures().fit_transform(raw)
        altered = raw.copy()
        altered["ID"] = -10000 - altered.ID
        altered["Response"] = 1 - altered.Response
        transformed = CustomerFeatures().fit_transform(altered)
        pd.testing.assert_frame_equal(features, transformed)
        self.assertNotIn("ID", features.columns)
        self.assertNotIn("Response", features.columns)
        self.assertTrue(features.loc[raw.Income.isna(), "Income"].isna().all())
        self.assertTrue((features.loc[raw.Income.isna(), "Income_missing"] == 1).all())
        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_saved_bundle_reproduces_scores_decisions_and_missing_income(self):
        example = self.X.iloc[:12].copy()
        before = predict_bundle(self.bundle, example)
        with tempfile.TemporaryDirectory(prefix="capstone-pipeline-test-") as tmp:
            path = Path(tmp) / "bundle.joblib"
            joblib.dump(self.bundle, path)
            loaded = joblib.load(path)
            after = predict_bundle(loaded, example)
        np.testing.assert_allclose(before.score, after.score, rtol=0, atol=0)
        np.testing.assert_array_equal(before.contact, after.contact)
        self.assertTrue(np.isfinite(after.score).all())
        self.assertTrue(after.score.between(0, 1).all())

    def test_inference_rejects_future_date_and_ineligible_population(self):
        future = self.X.iloc[:1].copy()
        future["Dt_Customer"] = pd.Timestamp("2015-01-01")
        with self.assertRaisesRegex(ValueError, "historical reference date"):
            predict_bundle(self.bundle, future)
        too_old = self.X.iloc[:1].copy()
        too_old["Year_Birth"] = 1900
        with self.assertRaisesRegex(ValueError, "age eligibility"):
            predict_bundle(self.bundle, too_old)
        invalid_income = self.X.iloc[:1].copy()
        invalid_income["Income"] = 666666
        with self.assertRaisesRegex(ValueError, "income eligibility"):
            predict_bundle(self.bundle, invalid_income)


if __name__ == "__main__":
    unittest.main()
