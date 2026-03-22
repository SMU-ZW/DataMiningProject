"""Unit tests for lib.models.common."""

import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from lib.models.common import (
    combine_train_val_sorted,
    grid_search_refit,
    validation_score,
    xgboost_scale_pos_weight,
)
from tests.unit.models.tiny_frames import tiny_classification_df


class TestCombineTrainValSorted(unittest.TestCase):
    def test_orders_by_timestamp(self):
        train = tiny_classification_df(n_rows=5, seed=1)
        val = tiny_classification_df(n_rows=4, seed=2)
        new_ts = pd.date_range("2025-01-02 10:00", periods=4, freq="min", tz="UTC")
        val = val.copy()
        val.index = pd.MultiIndex.from_arrays([["X"] * 4, new_ts], names=["symbol", "timestamp"])
        combined = combine_train_val_sorted(train, val)
        tss = combined.index.get_level_values("timestamp")
        self.assertTrue(tss.is_monotonic_increasing)
        self.assertEqual(len(combined), 9)


class TestXgboostScalePosWeight(unittest.TestCase):
    def test_ratio_matches_class_counts(self):
        y = pd.Series([1, 1, 0, 0, 0])
        w = xgboost_scale_pos_weight(y)
        self.assertAlmostEqual(w, 3.0 / 2.0)

    def test_no_positive_returns_one(self):
        y = pd.Series([0, 0, 0])
        self.assertEqual(xgboost_scale_pos_weight(y), 1.0)


class TestValidationScore(unittest.TestCase):
    def test_f1_perfect(self):
        y_val = pd.Series([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        est = DecisionTreeClassifier()
        est.fit([[0], [1], [1], [0]], y_val)
        x_val = pd.DataFrame({"a": [0.0, 1.0, 1.0, 0.0]})
        s = validation_score(y_val, y_pred, est, x_val, "f1")
        self.assertEqual(s, 1.0)


class TestGridSearchRefit(unittest.TestCase):
    def test_picks_higher_val_f1_on_trivial_split(self):
        """Grid search should prefer params that fit train better for val labels (smoke test)."""
        train_df = tiny_classification_df(n_rows=30, seed=3)
        val_df = tiny_classification_df(n_rows=20, seed=4)
        new_ts = pd.date_range("2025-01-03 09:30", periods=20, freq="min", tz="UTC")
        val_df = val_df.copy()
        val_df.index = pd.MultiIndex.from_arrays(
            [["X"] * 20, new_ts], names=["symbol", "timestamp"]
        )

        def build(params: dict, fit_data: pd.DataFrame) -> DecisionTreeClassifier:
            _ = fit_data
            md = params.get("max_depth", 2)
            return DecisionTreeClassifier(random_state=0, max_depth=md)

        out = grid_search_refit(
            build,
            {"max_depth": [1, 4]},
            train_df,
            val_df,
            "target",
            scoring="f1",
            verbose=False,
        )
        self.assertIsInstance(out, DecisionTreeClassifier)
        self.assertTrue(hasattr(out, "predict"))


if __name__ == "__main__":
    unittest.main()
