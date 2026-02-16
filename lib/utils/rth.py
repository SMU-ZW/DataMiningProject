"""
Regular trading hours (RTH) utilities using pandas_market_calendars schedules.
"""

from functools import reduce
from typing import Optional

import pandas as pd


def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return ts as timezone-aware UTC."""
    if ts.tz is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def rth_timestamps_from_schedule(
    schedule: pd.DataFrame,
    freq: pd.Timedelta,
    *,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DatetimeIndex:
    """
    Build RTH timestamps at the given frequency from a market schedule.

    For each schedule row (trading session), generates a date_range from
    market_open to market_close at `freq`, then unions all sessions and
    returns a UTC DatetimeIndex. Optionally clips to [start_ts, end_ts].

    Args:
        schedule: DataFrame with market_open and market_close columns (from
            e.g. nyse.schedule()).
        freq: Bar frequency (e.g. pd.Timedelta(minutes=1)).
        start_ts: If provided with end_ts, clip result to this range (UTC).
        end_ts: If provided with start_ts, clip result to this range (UTC).

    Returns:
        DatetimeIndex in UTC, sorted.
    """
    if schedule.empty:
        return pd.DatetimeIndex([])
    out = []
    for _, row in schedule.iterrows():
        session_open = row['market_open']
        session_close = row['market_close']
        session_ts = pd.date_range(
            start=session_open,
            end=session_close,
            freq=freq,
            inclusive='both',
        )
        out.append(session_ts)
    combined = reduce(lambda a, b: a.union(b), out)
    if combined.tz is not None:
        combined = combined.tz_convert('UTC')
    else:
        combined = combined.tz_localize('UTC')
    if start_ts is not None and end_ts is not None:
        start_ts = to_utc(start_ts)
        end_ts = to_utc(end_ts)
        mask = (combined >= start_ts) & (combined <= end_ts)
        combined = combined[mask]
    return combined
