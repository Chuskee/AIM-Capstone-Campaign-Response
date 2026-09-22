"""Checks for outcome exclusion, meaningful feature definitions and serialization."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.segmentation import FEATURES, SegmentFeatures, make_segment_preprocessor


def sample_customers():
    frame = pd.DataFrame({
        "Income": [30000.0, 60000.0, np.nan, 90000.0],
        "Year_Birth": [1980, 1970, 1990, 1960],
        "Dt_Customer": ["30-06-2014", "29-06-2014", "01-06-2014", "01-01-2014"],
        "Recency": [0, 10, 50, 90], "Kidhome": [1, 0, 0, 0], "Teenhome": [0, 1, 0, 0],
        "MntWines": [0, 50, 10, 100], "MntFruits": [0, 0, 0, 0],
        "MntMeatProducts": [0, 50, 10, 100], "MntFishProducts": [0, 0, 0, 0],
        "MntSweetProducts": [0, 0, 0, 0], "MntGoldProds": [0, 0, 0, 0],
        "NumWebPurchases": [0, 2, 1, 3], "NumCatalogPurchases": [0, 1, 0, 3],
        "NumStorePurchases": [0, 1, 1, 6], "NumDealsPurchases": [0, 1, 1, 2],
        "NumWebVisitsMonth": [1, 3, 2, 4], "Response": [0, 1, 0, 1],
        "AcceptedCmp1": [0, 1, 0, 1],
    })
    return frame


class SegmentationTests(unittest.TestCase):
    def test_feature_definitions_and_zero_denominators(self):
        features = SegmentFeatures().fit_transform(sample_customers())
        self.assertEqual(list(features), list(FEATURES))
        self.assertEqual(features.loc[0, "Age_2014"], 34)
        self.assertEqual(features.loc[0, "Tenure_Days"], 0)
        self.assertEqual(features.loc[1, "Tenure_Days"], 1)
        self.assertEqual(features.loc[0, "Wine_Spend_Share"], 0)
        self.assertEqual(features.loc[1, "Wine_Spend_Share"], 0.5)
        self.assertEqual(features.loc[1, "Web_Purchase_Share"], 0.5)
        self.assertEqual(features.loc[1, "Catalog_Purchase_Share"], 0.25)
        self.assertEqual(features.loc[1, "Children_At_Home"], 1)

    def test_outcomes_never_change_cluster_inputs(self):
        original = sample_customers()
        changed = original.copy()
        changed["Response"] = 1 - changed["Response"]
        changed["AcceptedCmp1"] = 1 - changed["AcceptedCmp1"]
        prep = make_segment_preprocessor().fit(original)
        np.testing.assert_array_equal(prep.transform(original), prep.transform(changed))
        self.assertTrue(np.isfinite(prep.transform(original)).all())
        self.assertTrue(pd.isna(original.loc[2, "Income"]))
        dropped = original.drop(columns=["Response", "AcceptedCmp1"])
        np.testing.assert_array_equal(prep.transform(original), prep.transform(dropped))

    def test_saved_pipeline_predicts_without_outcome_columns(self):
        frame = sample_customers()
        prep = make_segment_preprocessor()
        model = Pipeline(prep.steps + [("cluster", KMeans(n_clusters=2, random_state=42, n_init=5))])
        model.fit(frame)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "segments.joblib"
            joblib.dump(model, file)
            loaded = joblib.load(file)
            np.testing.assert_array_equal(
                model.predict(frame),
                loaded.predict(frame.drop(columns=["Response", "AcceptedCmp1"])),
            )


if __name__ == "__main__":
    unittest.main()
