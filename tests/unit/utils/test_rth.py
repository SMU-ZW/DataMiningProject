"""
Unit tests for RTH (regular trading hours) utilities.
"""

import unittest

import pandas as pd

from lib.utils.rth import rth_timestamps_from_schedule, to_utc


class TestToUtc(unittest.TestCase):
    """Test to_utc."""

    def test_naive_timestamp_localized_to_utc(self):
        """Naive timestamp is localized to UTC."""
        ts = pd.Timestamp('2024-01-02 14:30:00')
        result = to_utc(ts)
        self.assertIsNotNone(result.tz)
        self.assertEqual(str(result.tz), 'UTC')
        self.assertEqual(result.hour, 14)
        self.assertEqual(result.minute, 30)

    def test_aware_timestamp_converted_to_utc(self):
        """Timezone-aware timestamp is converted to UTC."""
        ts = pd.Timestamp('2024-01-02 09:30:00', tz='America/New_York')
        result = to_utc(ts)
        self.assertEqual(str(result.tz), 'UTC')
        # 9:30 Eastern = 14:30 UTC (EST)
        self.assertEqual(result.hour, 14)
        self.assertEqual(result.minute, 30)


class TestRthTimestampsFromSchedule(unittest.TestCase):
    """Test rth_timestamps_from_schedule."""

    def _schedule_one_session(self, open_ts, close_ts):
        """Build a single-row schedule DataFrame with market_open, market_close."""
        return pd.DataFrame({
            'market_open': [open_ts],
            'market_close': [close_ts],
        })

    def test_empty_schedule_returns_empty_index(self):
        """Empty schedule returns empty DatetimeIndex."""
        schedule = pd.DataFrame(columns=['market_open', 'market_close'])
        result = rth_timestamps_from_schedule(
            schedule, pd.Timedelta(minutes=1)
        )
        self.assertIsInstance(result, pd.DatetimeIndex)
        self.assertEqual(len(result), 0)

    def test_single_session_returns_bars_at_freq(self):
        """Single session yields timestamps at the given frequency."""
        open_ts = pd.Timestamp('2024-01-02 09:30', tz='America/New_York')
        close_ts = pd.Timestamp('2024-01-02 09:33', tz='America/New_York')
        schedule = self._schedule_one_session(open_ts, close_ts)
        result = rth_timestamps_from_schedule(
            schedule, pd.Timedelta(minutes=1)
        )
        self.assertIsInstance(result, pd.DatetimeIndex)
        self.assertEqual(len(result), 4)  # 09:30, 09:31, 09:32, 09:33
        self.assertIsNotNone(result.tz)
        self.assertEqual(str(result.tz), 'UTC')

    def test_two_sessions_are_unioned(self):
        """Two sessions produce unioned timestamps."""
        open1 = pd.Timestamp('2024-01-02 09:30', tz='America/New_York')
        close1 = pd.Timestamp('2024-01-02 09:31', tz='America/New_York')
        open2 = pd.Timestamp('2024-01-03 09:30', tz='America/New_York')
        close2 = pd.Timestamp('2024-01-03 09:31', tz='America/New_York')
        schedule = pd.DataFrame({
            'market_open': [open1, open2],
            'market_close': [close1, close2],
        })
        result = rth_timestamps_from_schedule(
            schedule, pd.Timedelta(minutes=1)
        )
        self.assertEqual(len(result), 4)  # 2 bars per day × 2 days

    def test_clip_to_start_ts_end_ts(self):
        """Optional start_ts and end_ts clip the result."""
        open_ts = pd.Timestamp('2024-01-02 09:30', tz='America/New_York')
        close_ts = pd.Timestamp('2024-01-02 10:00', tz='America/New_York')
        schedule = self._schedule_one_session(open_ts, close_ts)
        # Clip to 09:35–09:39 UTC (narrow window)
        start_ts = pd.Timestamp('2024-01-02 14:35', tz='UTC')  # 09:35 Eastern
        end_ts = pd.Timestamp('2024-01-02 14:39', tz='UTC')
        result = rth_timestamps_from_schedule(
            schedule,
            pd.Timedelta(minutes=1),
            start_ts=start_ts,
            end_ts=end_ts,
        )
        self.assertEqual(len(result), 5)  # 14:35, 14:36, 14:37, 14:38, 14:39 UTC
        self.assertGreaterEqual(result.min(), start_ts)
        self.assertLessEqual(result.max(), end_ts)

    def test_clip_with_naive_timestamps_normalized_to_utc(self):
        """start_ts/end_ts as naive are normalized to UTC for clipping."""
        open_ts = pd.Timestamp('2024-01-02 09:30', tz='America/New_York')
        close_ts = pd.Timestamp('2024-01-02 09:35', tz='America/New_York')
        schedule = self._schedule_one_session(open_ts, close_ts)
        start_ts = pd.Timestamp('2024-01-02 14:32')  # naive, 14:32 UTC
        end_ts = pd.Timestamp('2024-01-02 14:34')
        result = rth_timestamps_from_schedule(
            schedule,
            pd.Timedelta(minutes=1),
            start_ts=start_ts,
            end_ts=end_ts,
        )
        self.assertEqual(len(result), 3)  # 14:32, 14:33, 14:34
        self.assertEqual(str(result.tz), 'UTC')
