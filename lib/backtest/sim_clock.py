"""
Clock that advances with the backtest timeline (bar timestamps from the data).
"""

from datetime import datetime
from typing import List

import pandas as pd

from lib.framework.clock import Clock


class SimClock(Clock):
    """Clock that yields sorted unique timestamps from a DataFrame or list."""

    def __init__(self, timestamps: List[datetime] | pd.DatetimeIndex) -> None:
        """
        Args:
            timestamps: Sorted list of bar timestamps (e.g. from DataFrame index).
        """
        if hasattr(timestamps, "tolist"):
            timestamps = timestamps.tolist()
        self._times = sorted(set(timestamps))
        self._index = -1

    @property
    def current_time(self) -> datetime:
        """Current bar time."""
        if self._index < 0 or self._index >= len(self._times):
            raise RuntimeError("Clock not advanced; call advance() first")
        return self._times[self._index]

    def advance(self) -> bool:
        """Advance to the next bar. Returns True if there is a next bar."""
        self._index += 1
        return self._index < len(self._times)
