"""Unit tests for lib.models.naive_bayes."""

import unittest

import pandas as pd

from lib.models.naive_bayes import train_naive_bayes
from tests.unit.models.tiny_frames import tiny_classification_df


class TestTrainNaiveBayes(unittest.TestCase):
    def test_grid_search_returns_fitted_model(self):
        train_df = tiny_classification_df(n_rows=35, seed=5)
        val_df = tiny_classification_df(n_rows=15, seed=6)
        new_ts = pd.date_range("2025-01-04 09:30", periods=15, freq="min", tz="UTC")
        val_df = val_df.copy()
        val_df.index = pd.MultiIndex.from_arrays(
            [["X"] * 15, new_ts], names=["symbol", "timestamp"]
        )
        clf = train_naive_bayes(
            train_df,
            val_df,
            param_grid={"var_smoothing": [1e-9, 1e-6]},
            verbose=False,
        )
        preds = clf.predict(train_df.drop(columns=["target"]))
        self.assertEqual(len(preds), len(train_df))


if __name__ == "__main__":
    unittest.main()
