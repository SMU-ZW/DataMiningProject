"""Unit tests for lib.models.xgboost."""

import unittest

import pandas as pd

from tests.unit.models.tiny_frames import tiny_classification_df

try:
    import xgboost  # noqa: F401

    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False


@unittest.skipUnless(HAVE_XGBOOST, "xgboost not installed")
class TestTrainXgboost(unittest.TestCase):
    def test_grid_search_returns_fitted_model(self):
        from lib.models.xgboost import train_xgboost

        train_df = tiny_classification_df(n_rows=30, seed=17)
        val_df = tiny_classification_df(n_rows=12, seed=18)
        new_ts = pd.date_range("2025-01-09 09:30", periods=12, freq="min", tz="UTC")
        val_df = val_df.copy()
        val_df.index = pd.MultiIndex.from_arrays(
            [["X"] * 12, new_ts], names=["symbol", "timestamp"]
        )
        clf = train_xgboost(
            train_df,
            val_df,
            param_grid={
                "n_estimators": [10, 20],
                "max_depth": [2, 3],
                "learning_rate": [0.1],
            },
            verbose=False,
        )
        preds = clf.predict(train_df.drop(columns=["target"]))
        self.assertEqual(len(preds), len(train_df))


if __name__ == "__main__":
    unittest.main()
