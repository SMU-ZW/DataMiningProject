"""Unit tests for lib.models.decision_tree."""

import unittest

import pandas as pd

from lib.models.decision_tree import train_decision_tree
from tests.unit.models.tiny_frames import tiny_classification_df


class TestTrainDecisionTreeNoGrid(unittest.TestCase):
    def test_single_fit_on_train_plus_val(self):
        train_df = tiny_classification_df(n_rows=25, seed=7)
        val_df = tiny_classification_df(n_rows=10, seed=8)
        new_ts = pd.date_range("2025-01-05 09:30", periods=10, freq="min", tz="UTC")
        val_df = val_df.copy()
        val_df.index = pd.MultiIndex.from_arrays(
            [["X"] * 10, new_ts], names=["symbol", "timestamp"]
        )
        clf = train_decision_tree(train_df, val_df, param_grid=None, max_depth=3)
        features = pd.concat([train_df, val_df]).sort_index(level="timestamp").drop(
            columns=["target"]
        )
        preds = clf.predict(features)
        self.assertEqual(len(preds), 35)


if __name__ == "__main__":
    unittest.main()
