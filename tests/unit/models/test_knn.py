"""Unit tests for lib.models.knn."""

import unittest

import pandas as pd

from lib.models.knn import train_knn
from tests.unit.models.tiny_frames import tiny_classification_df


class TestTrainKnn(unittest.TestCase):
    def test_grid_search_returns_fitted_model(self):
        train_df = tiny_classification_df(n_rows=32, seed=15)
        val_df = tiny_classification_df(n_rows=10, seed=16)
        new_ts = pd.date_range("2025-01-08 09:30", periods=10, freq="min", tz="UTC")
        val_df = val_df.copy()
        val_df.index = pd.MultiIndex.from_arrays(
            [["X"] * 10, new_ts], names=["symbol", "timestamp"]
        )
        clf = train_knn(
            train_df,
            val_df,
            param_grid={"n_neighbors": [3, 5], "weights": ["uniform"]},
            verbose=False,
        )
        preds = clf.predict(train_df.drop(columns=["target"]))
        self.assertEqual(len(preds), len(train_df))


if __name__ == "__main__":
    unittest.main()
