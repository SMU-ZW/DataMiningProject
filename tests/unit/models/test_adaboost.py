"""Unit tests for lib.models.adaboost."""

import unittest

import pandas as pd

from lib.models.adaboost import train_adaboost
from tests.unit.models.tiny_frames import tiny_classification_df


class TestTrainAdaboost(unittest.TestCase):
    def test_grid_search_returns_fitted_model(self):
        train_df = tiny_classification_df(n_rows=28, seed=13)
        val_df = tiny_classification_df(n_rows=14, seed=14)
        new_ts = pd.date_range("2025-01-07 09:30", periods=14, freq="min", tz="UTC")
        val_df = val_df.copy()
        val_df.index = pd.MultiIndex.from_arrays(
            [["X"] * 14, new_ts], names=["symbol", "timestamp"]
        )
        clf = train_adaboost(
            train_df,
            val_df,
            param_grid={"n_estimators": [20, 40], "learning_rate": [0.5, 1.0]},
            verbose=False,
        )
        preds = clf.predict(train_df.drop(columns=["target"]))
        self.assertEqual(len(preds), len(train_df))


if __name__ == "__main__":
    unittest.main()
