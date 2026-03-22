"""Unit tests for lib.models.random_forest."""

import unittest

import pandas as pd

from lib.models.random_forest import train_forest
from tests.unit.models.tiny_frames import tiny_classification_df


class TestTrainRandomForest(unittest.TestCase):
    def test_grid_search_returns_fitted_model(self):
        train_df = tiny_classification_df(n_rows=30, seed=11)
        val_df = tiny_classification_df(n_rows=12, seed=12)
        new_ts = pd.date_range("2025-01-06 09:30", periods=12, freq="min", tz="UTC")
        val_df = val_df.copy()
        val_df.index = pd.MultiIndex.from_arrays(
            [["X"] * 12, new_ts], names=["symbol", "timestamp"]
        )
        clf = train_forest(
            train_df,
            val_df,
            param_grid={"n_estimators": [10, 20], "max_depth": [2, 3]},
            verbose=False,
        )
        preds = clf.predict(train_df.drop(columns=["target"]))
        self.assertEqual(len(preds), len(train_df))


if __name__ == "__main__":
    unittest.main()
