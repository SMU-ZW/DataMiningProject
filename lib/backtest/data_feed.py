"""
DataFrame-backed data feed: returns bar(s) at a given time for backtest.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from lib.framework.data_feed import DataFeed


class DataFrameDataFeed(DataFeed):
    """Data feed that slices a preloaded MultiIndex (symbol, timestamp) DataFrame."""

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Args:
            data: DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns.
        """
        self._data = data.sort_index() if isinstance(data.index, pd.MultiIndex) and not data.empty else data

    def get_bars(
        self,
        current_time: datetime,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return rows at current_time, optionally for a single symbol."""
        if self._data.empty:
            return pd.DataFrame()
        if not isinstance(self._data.index, pd.MultiIndex):
            return self._data
        idx = self._data.index
        timestamps = idx.get_level_values("timestamp")
        ts = pd.Timestamp(current_time)
        if hasattr(timestamps, "tz") and timestamps.tz is not None:
            ts = ts.tz_localize(timestamps.tz) if ts.tz is None else ts.tz_convert(timestamps.tz)
        mask = timestamps == ts
        if symbol is not None:
            sym_level = idx.get_level_values("symbol")
            mask = mask & (sym_level == symbol)
        out = self._data.loc[mask]
        if isinstance(out, pd.Series):
            return pd.DataFrame([out])
        return out.copy() if isinstance(out, pd.DataFrame) else pd.DataFrame()
