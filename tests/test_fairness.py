"""Tests for frozen policy application, undefined rates and audit arithmetic."""

import copy
import json
import unittest

import numpy as np
import pandas as pd

from src.fairness import (apply_policy, audit_fairness, evaluate_predictions,
                          fit_policy, make_sensitive, wilson_interval)


class FairnessTests(unittest.TestCase):
    def test_sensitive_groups_are_historical_and_missing_is_explicit(self):
        raw = pd.DataFrame({"Year_Birth": [1980, 1979, 1964, 1949, np.nan],
                            "Income": [10, 25, 50, 1000, np.nan],
                            "Education": ["Basic", "Master", "PhD", "Graduation", None],
                            "Kidhome": [0, 1, 0, 0, np.nan], "Teenhome": [0, 0, 1, 0, 0],
                            "Marital_Status": ["Married", "Together", "Single", "Widow", None]})
        groups = make_sensitive(raw, [0, 25, 50, 75, 100])
        self.assertEqual(groups.age_band.tolist(), ["Under 35", "35–49", "50–64", "65+", "Unknown"])
        self.assertEqual(groups.income_quartile.tolist(), ["Q1", "Q1", "Q2", "Q4", "Unknown"])
        self.assertEqual(groups.children_at_home.tolist(), ["No children at home", "Children at home", "Children at home", "No children at home", "Unknown"])
        self.assertEqual(groups.household.tolist(), ["Partnered", "Partnered", "Not partnered", "Not partnered", "Unknown"])

    def test_unknown_household_labels_are_not_treated_as_unpartnered(self):
        raw = pd.DataFrame({"Year_Birth": [1980] * 5, "Income": [30] * 5,
                            "Education": ["Basic"] * 5, "Kidhome": [0] * 5,
                            "Teenhome": [0] * 5,
                            "Marital_Status": ["Unknown", "Absurd", "YOLO", None, "Alone"]})
        groups = make_sensitive(raw, [25, 50, 75])
        self.assertEqual(groups.household.tolist(), ["Unknown", "Unknown", "Unknown", "Unknown", "Not partnered"])

    def test_audit_matches_hand_calculated_confusion_rates(self):
        y = [1, 1, 0, 0, 1, 1, 0, 0]
        pred = [1, 0, 1, 0, 1, 1, 0, 0]
        audit = audit_fairness(y, pred, pd.DataFrame({"axis": ["A"] * 4 + ["B"] * 4}))
        a, b = audit["groups"]
        self.assertEqual((a["tp"], a["fp"], a["tn"], a["fn"]), (1, 1, 1, 1))
        self.assertEqual((b["tpr"], b["fpr"], b["precision"]), (1, 0, 1))
        summary = audit["summary"][0]
        self.assertEqual(summary["demographic_parity_gap"], 0)
        self.assertEqual(summary["disparate_impact_ratio"], 1)
        self.assertEqual(summary["equalised_odds_gap"], 0.5)
        self.assertLess(a["selection_wilson_low"], 0.5)
        self.assertGreater(a["selection_wilson_high"], 0.5)

    def test_no_positives_and_no_selection_do_not_claim_zero_error_gap(self):
        audit = audit_fairness([0, 0, 0, 0], [0, 0, 0, 0], pd.DataFrame({"axis": ["A", "A", "B", "B"]}))
        self.assertIsNone(audit["groups"][0]["tpr"])
        self.assertIsNone(audit["groups"][0]["tpr_wilson_low"])
        self.assertIsNone(audit["groups"][0]["precision"])
        self.assertIsNone(audit["summary"][0]["equalised_odds_gap"])
        self.assertIsNone(audit["summary"][0]["disparate_impact_ratio"])
        self.assertEqual(audit["summary"][0]["demographic_parity_gap"], 0)
        json.dumps(audit, allow_nan=False)

    def test_policy_chooses_smallest_successful_alpha_and_is_frozen(self):
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        scores = np.array([.95, .9, .8, .7, .6, .4, .2, .1])
        groups = np.array(["A"] * 4 + ["B"] * 4)
        policy = fit_policy(y, scores, groups)
        feasible = [row for row in policy["validation_grid"]
                    if row["selection_rate_ratio"] is not None and row["selection_rate_ratio"] >= .8]
        self.assertTrue(feasible)
        self.assertEqual(policy["alpha"], feasible[0]["alpha"])
        before = copy.deepcopy(policy)
        test_scores = [.99, .05, .85, .15]
        first = apply_policy(test_scores, ["A", "A", "B", "B"], policy)
        second = apply_policy(test_scores, ["A", "A", "B", "B"], policy)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(before, policy)
        self.assertEqual(policy["fit_partition"], "validation")
        json.dumps(policy, allow_nan=False)

    def test_unseen_groups_raise_instead_of_refitting_or_falling_back(self):
        policy = fit_policy([0, 1, 0, 1], [.1, .9, .2, .8], ["A", "A", "B", "B"])
        with self.assertRaisesRegex(ValueError, "unseen group"):
            apply_policy([.5], ["C"], policy)

    def test_infeasible_policy_reports_global_fallback(self):
        policy = fit_policy([0, 0, 0, 0], [.2, .4, .6, .8], ["A", "A", "B", "B"])
        self.assertEqual(policy["status"], "target_not_achieved_global_fallback")
        self.assertEqual(policy["alpha"], 0)
        self.assertEqual(int(apply_policy([.2, .4, .6, .8], ["A", "A", "B", "B"], policy).sum()), 0)
        json.dumps(policy, allow_nan=False)

    def test_evaluation_uses_same_policy_confusion_counts_for_profit(self):
        result = evaluate_predictions([1, 0, 1, 0], [.9, .8, .3, .1], [1, 1, 0, 0])
        self.assertEqual(result["observed_label_profit_proxy"], 11 * 1 - 3 * 2)
        self.assertEqual(result["contact_fraction"], .5)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        json.dumps(result, allow_nan=False)
        empty_class = evaluate_predictions([0, 0], [.1, .2], [0, 0])
        self.assertIsNone(empty_class["roc_auc"])
        self.assertIsNone(empty_class["average_precision"])
        self.assertIsNone(empty_class["observed_label_roi_proxy"])
        json.dumps(empty_class, allow_nan=False)

    def test_wilson_boundaries_and_invalid_inputs(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))
        low, high = wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0)
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)
        with self.assertRaises(ValueError):
            fit_policy([0, 1], [.1], ["A", "B"])
        with self.assertRaises(ValueError):
            apply_policy([float("nan")], ["A"], {"group_thresholds": {"A": .5}})


if __name__ == "__main__":
    unittest.main()
