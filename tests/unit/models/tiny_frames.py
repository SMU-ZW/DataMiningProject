"""Small synthetic MultiIndex frames for lib.models unit tests."""

import numpy as np
import pandas as pd


def tiny_classification_df(*, n_rows: int = 40, seed: int = 0) -> pd.DataFrame:
    """MultiIndex (symbol, timestamp), binary target, two numeric features."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-02 09:30", periods=n_rows, freq="min", tz="UTC")
    index = pd.MultiIndex.from_arrays([["X"] * n_rows, ts], names=["symbol", "timestamp"])
    x1 = rng.normal(size=n_rows)
    x2 = rng.normal(size=n_rows)
    target = ((x1 + x2) > 0).astype(np.int64)
    return pd.DataFrame({"f1": x1, "f2": x2, "target": target}, index=index)
