"""Unit tests for orion experiment feature helpers (elib)."""

import unittest

import numpy as np
import pandas as pd

from experiments.orion.elib import (
    add_feature_atr,
    add_feature_close_vwap_pct_diff,
    add_feature_day_of_week,
    add_feature_trade_count_zscore,
    add_feature_volume_and_trade_count_zscore,
    add_feature_volume_zscore,
)


def _ohlc(symbol: str, rows):
    """
    Build DataFrame with MultiIndex (symbol, timestamp) and OHLC columns.

    rows: sequence of (timestamp, open, high, low, close) per bar.
    """
    if not rows:
        index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=index)
    timestamps = pd.to_datetime([r[0] for r in rows])
    opens = [r[1] for r in rows]
    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    closes = [r[4] for r in rows]
    index = pd.MultiIndex.from_arrays(
        [[symbol] * len(rows), timestamps],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )


def _ohlcv(symbol: str, rows):
    """
    Build DataFrame with MultiIndex (symbol, timestamp), OHLC, and volume.

    rows: sequence of (timestamp, open, high, low, close, volume) per bar.
    """
    if not rows:
        index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=index,
        )
    timestamps = pd.to_datetime([r[0] for r in rows])
    opens = [r[1] for r in rows]
    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    closes = [r[4] for r in rows]
    vols = [r[5] for r in rows]
    index = pd.MultiIndex.from_arrays(
        [[symbol] * len(rows), timestamps],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=index,
    )


def _ohlcv_tc(symbol: str, rows):
    """
    Build DataFrame with MultiIndex (symbol, timestamp), OHLC, volume, and trade_count.

    rows: sequence of (timestamp, open, high, low, close, volume, trade_count) per bar.
    """
    if not rows:
        index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "trade_count"],
            index=index,
        )
    timestamps = pd.to_datetime([r[0] for r in rows])
    opens = [r[1] for r in rows]
    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    closes = [r[4] for r in rows]
    vols = [r[5] for r in rows]
    tcs = [r[6] for r in rows]
    index = pd.MultiIndex.from_arrays(
        [[symbol] * len(rows), timestamps],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "trade_count": tcs,
        },
        index=index,
    )


def _ohlcv_vwap(symbol: str, rows):
    """
    Build DataFrame with MultiIndex (symbol, timestamp), OHLC, and vwap.

    rows: sequence of (timestamp, open, high, low, close, vwap) per bar.
    timestamp can be a string (e.g. "2025-01-02 09:31") or datetime-like.
    """
    if not rows:
        index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "vwap"],
            index=index,
        )
    timestamps = pd.to_datetime([r[0] for r in rows])
    opens = [r[1] for r in rows]
    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    closes = [r[4] for r in rows]
    vwaps = [r[5] for r in rows]
    index = pd.MultiIndex.from_arrays(
        [[symbol] * len(rows), timestamps],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "vwap": vwaps},
        index=index,
    )


class TestAddFeatureCloseVwapPctDiff(unittest.TestCase):
    """Tests for add_feature_close_vwap_pct_diff."""

    def test_close_equals_vwap_is_zero(self):
        df = _ohlcv_vwap(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100.5, 100.5),
            ],
        )
        result = add_feature_close_vwap_pct_diff(df)
        self.assertIn("close_vwap_pct_diff", result.columns)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[0], 0.0)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[1], 0.0)

    def test_close_above_vwap(self):
        # (101 - 100) / 100 = 0.01
        df = _ohlcv_vwap(
            "AAPL",
            [("2025-01-02 09:31", 100, 102, 99, 101, 100)],
        )
        result = add_feature_close_vwap_pct_diff(df)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[0], 0.01)

    def test_close_below_vwap(self):
        # (99 - 100) / 100 = -0.01
        df = _ohlcv_vwap(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 98, 99, 100)],
        )
        result = add_feature_close_vwap_pct_diff(df)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[0], -0.01)

    def test_vwap_zero_no_division(self):
        df = _ohlcv_vwap(
            "AAPL",
            [("2025-01-02 09:31", 0, 0, 0, 100, 0)],
        )
        result = add_feature_close_vwap_pct_diff(df)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[0], 0.0)

    def test_multiple_days(self):
        df = _ohlcv_vwap(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 101, 99, 100, 100),
                ("2025-01-03 09:31", 200, 201, 199, 202, 200),
            ],
        )
        result = add_feature_close_vwap_pct_diff(df)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[0], 0.0)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[1], 0.01)

    def test_multiple_symbols_independent(self):
        df_aapl = _ohlcv_vwap(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 110, 100)],
        )
        df_googl = _ohlcv_vwap(
            "GOOGL",
            [("2025-01-02 09:31", 50, 51, 49, 45, 50)],
        )
        df = pd.concat([df_aapl, df_googl])
        result = add_feature_close_vwap_pct_diff(df)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[0], 0.1)
        self.assertEqual(result["close_vwap_pct_diff"].iloc[1], -0.1)

    def test_custom_column_name(self):
        df = _ohlcv_vwap(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 101, 100)],
        )
        result = add_feature_close_vwap_pct_diff(df, column_name="feat_vwap")
        self.assertIn("feat_vwap", result.columns)
        self.assertNotIn("close_vwap_pct_diff", result.columns)
        self.assertEqual(result["feat_vwap"].iloc[0], 0.01)


class TestAddFeatureAtr(unittest.TestCase):
    """Tests for add_feature_atr (Wilder ATR)."""

    def test_period_three_first_atr_is_mean_of_three_trs(self):
        # Bar0: TR = 101-99 = 2. Bar1: max(1,0.5,0.5)=1. Bar2: max(3,2,1)=3. ATR[2]=(2+1+3)/3=2.
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),
                ("2025-01-02 09:33", 100, 102, 99, 101),
            ],
        )
        result = add_feature_atr(df, [3])
        self.assertIn("atr_3", result.columns)
        self.assertEqual(result["atr_3"].iloc[0], 0.0)
        self.assertEqual(result["atr_3"].iloc[1], 0.0)
        self.assertEqual(result["atr_3"].iloc[2], 2.0)

    def test_wilder_smoothing_after_initial(self):
        # Same TRs as above; period=2: ATR[1]=(2+1)/2=1.5, then ATR[2]=(1.5*1+3)/2=2.25
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),
                ("2025-01-02 09:33", 100, 102, 99, 101),
            ],
        )
        result = add_feature_atr(df, [2])
        self.assertEqual(result["atr_2"].iloc[0], 0.0)
        self.assertEqual(result["atr_2"].iloc[1], 1.5)
        self.assertEqual(result["atr_2"].iloc[2], 2.25)

    def test_period_one_equals_true_range(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),
            ],
        )
        result = add_feature_atr(df, [1])
        self.assertEqual(result["atr_1"].iloc[0], 2.0)
        self.assertEqual(result["atr_1"].iloc[1], 1.0)

    def test_not_enough_bars_all_zero(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),
            ],
        )
        result = add_feature_atr(df, [5])
        self.assertEqual(result["atr_5"].iloc[0], 0.0)
        self.assertEqual(result["atr_5"].iloc[1], 0.0)

    def test_multiple_symbols_independent(self):
        df_aapl = _ohlc(
            "AAPL",
            [("2025-01-02 09:31", 100, 110, 90, 100)],
        )
        df_googl = _ohlc(
            "GOOGL",
            [("2025-01-02 09:31", 50, 60, 40, 50)],
        )
        df = pd.concat([df_aapl, df_googl])
        result = add_feature_atr(df, [1])
        self.assertEqual(result["atr_1"].iloc[0], 20.0)
        self.assertEqual(result["atr_1"].iloc[1], 20.0)

    def test_prev_close_spans_days(self):
        # Day1 last close 100; day2 bar uses prev close for TR
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 16:00", 100, 101, 99, 100),
                ("2025-01-03 09:31", 100, 105, 98, 102),
            ],
        )
        result = add_feature_atr(df, [1])
        self.assertEqual(result["atr_1"].iloc[0], 2.0)
        # TR = max(7, 5, 2) = 7
        self.assertEqual(result["atr_1"].iloc[1], 7.0)

    def test_custom_column_name(self):
        df = _ohlc(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 100)],
        )
        result = add_feature_atr(df, [1], column_name_fn=lambda p: "my_atr")
        self.assertIn("my_atr", result.columns)
        self.assertNotIn("atr_1", result.columns)
        self.assertEqual(result["my_atr"].iloc[0], 2.0)

    def test_empty_period_list_returns_unchanged(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),
            ],
        )
        result = add_feature_atr(df, [])
        self.assertEqual(list(result.columns), list(df.columns))
        pd.testing.assert_frame_equal(result[df.columns], df)

    def test_multiple_periods_match_sequential_calls(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),
                ("2025-01-02 09:33", 100, 102, 99, 101),
            ],
        )
        single = add_feature_atr(df.copy(), [2])
        single = add_feature_atr(single, [3])
        batch = add_feature_atr(df.copy(), [2, 3])
        pd.testing.assert_series_equal(single["atr_2"], batch["atr_2"])
        pd.testing.assert_series_equal(single["atr_3"], batch["atr_3"])

    def test_period_zero_raises(self):
        df = _ohlc("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100)])
        with self.assertRaises(ValueError):
            add_feature_atr(df, [0])


class TestAddFeatureVolumeZscore(unittest.TestCase):
    """Tests for add_feature_volume_zscore (within-day rolling z-score)."""

    def test_window_three_spike_on_fourth_bar(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100, 100),
                ("2025-01-02 09:33", 100, 101, 99, 100, 100),
                ("2025-01-02 09:34", 100, 101, 99, 100, 1000),
            ],
        )
        result = add_feature_volume_zscore(df, [3])
        self.assertIn("volume_zscore_3", result.columns)
        self.assertEqual(result["volume_zscore_3"].iloc[0], 0.0)
        self.assertEqual(result["volume_zscore_3"].iloc[1], 0.0)
        seg = np.array([100.0, 100.0, 1000.0])
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=0))
        expected = (1000.0 - m) / s
        self.assertAlmostEqual(float(result["volume_zscore_3"].iloc[3]), expected)

    def test_constant_volume_in_window_z_zero(self):
        # Third bar: window all 100 -> std 0 -> z 0
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100, 100),
                ("2025-01-02 09:33", 100, 101, 99, 100, 100),
            ],
        )
        result = add_feature_volume_zscore(df, [3])
        self.assertEqual(float(result["volume_zscore_3"].iloc[2]), 0.0)

    def test_new_day_resets_window(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 101, 99, 100, 1),
                ("2025-01-02 16:00", 100, 101, 99, 100, 2),
                ("2025-01-03 09:31", 100, 101, 99, 100, 10),
                ("2025-01-03 09:32", 100, 101, 99, 100, 20),
            ],
        )
        result = add_feature_volume_zscore(df, [2])
        self.assertEqual(result["volume_zscore_2"].iloc[2], 0.0)
        seg = np.array([10.0, 20.0])
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=0))
        self.assertAlmostEqual(float(result["volume_zscore_2"].iloc[3]), (20.0 - m) / s)

    def test_empty_window_list_returns_unchanged(self):
        df = _ohlcv(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 100, 100)],
        )
        result = add_feature_volume_zscore(df, [])
        self.assertEqual(list(result.columns), list(df.columns))
        pd.testing.assert_frame_equal(result[df.columns], df)

    def test_multiple_windows_match_sequential_calls(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 10),
                ("2025-01-02 09:32", 100, 101, 99, 100, 20),
                ("2025-01-02 09:33", 100, 101, 99, 100, 30),
            ],
        )
        single = add_feature_volume_zscore(df.copy(), [2])
        single = add_feature_volume_zscore(single, [3])
        batch = add_feature_volume_zscore(df.copy(), [2, 3])
        pd.testing.assert_series_equal(single["volume_zscore_2"], batch["volume_zscore_2"])
        pd.testing.assert_series_equal(single["volume_zscore_3"], batch["volume_zscore_3"])

    def test_custom_column_name_fn(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 10),
                ("2025-01-02 09:32", 100, 101, 99, 100, 20),
            ],
        )
        result = add_feature_volume_zscore(df, [2], column_name_fn=lambda w: f"vz_{w}")
        self.assertIn("vz_2", result.columns)
        self.assertNotIn("volume_zscore_2", result.columns)

    def test_window_zero_raises(self):
        df = _ohlcv("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100, 100)])
        with self.assertRaises(ValueError):
            add_feature_volume_zscore(df, [0])


class TestAddFeatureTradeCountZscore(unittest.TestCase):
    """Tests for add_feature_trade_count_zscore (within-day rolling z-score)."""

    def test_uses_trade_count_not_volume(self):
        # Volume flat; trade_count spikes like volume z-score spike test.
        df = _ohlcv_tc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 5000, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100, 5000, 100),
                ("2025-01-02 09:33", 100, 101, 99, 100, 5000, 100),
                ("2025-01-02 09:34", 100, 101, 99, 100, 5000, 1000),
            ],
        )
        result = add_feature_trade_count_zscore(df, [3])
        self.assertIn("trade_count_zscore_3", result.columns)
        seg = np.array([100.0, 100.0, 1000.0])
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=0))
        expected = (1000.0 - m) / s
        self.assertAlmostEqual(float(result["trade_count_zscore_3"].iloc[3]), expected)

    def test_constant_trade_count_in_window_z_zero(self):
        df = _ohlcv_tc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 1, 50),
                ("2025-01-02 09:32", 100, 101, 99, 100, 2, 50),
                ("2025-01-02 09:33", 100, 101, 99, 100, 3, 50),
            ],
        )
        result = add_feature_trade_count_zscore(df, [3])
        self.assertEqual(float(result["trade_count_zscore_3"].iloc[2]), 0.0)

    def test_new_day_resets_window(self):
        df = _ohlcv_tc(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 101, 99, 100, 100, 1),
                ("2025-01-02 16:00", 100, 101, 99, 100, 100, 2),
                ("2025-01-03 09:31", 100, 101, 99, 100, 100, 10),
                ("2025-01-03 09:32", 100, 101, 99, 100, 100, 20),
            ],
        )
        result = add_feature_trade_count_zscore(df, [2])
        self.assertEqual(result["trade_count_zscore_2"].iloc[2], 0.0)
        seg = np.array([10.0, 20.0])
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=0))
        self.assertAlmostEqual(float(result["trade_count_zscore_2"].iloc[3]), (20.0 - m) / s)

    def test_empty_window_list_returns_unchanged(self):
        df = _ohlcv_tc(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 100, 1, 100)],
        )
        result = add_feature_trade_count_zscore(df, [])
        self.assertEqual(list(result.columns), list(df.columns))
        pd.testing.assert_frame_equal(result[df.columns], df)

    def test_multiple_windows_match_sequential_calls(self):
        df = _ohlcv_tc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 1, 10),
                ("2025-01-02 09:32", 100, 101, 99, 100, 1, 20),
                ("2025-01-02 09:33", 100, 101, 99, 100, 1, 30),
            ],
        )
        single = add_feature_trade_count_zscore(df.copy(), [2])
        single = add_feature_trade_count_zscore(single, [3])
        batch = add_feature_trade_count_zscore(df.copy(), [2, 3])
        pd.testing.assert_series_equal(
            single["trade_count_zscore_2"], batch["trade_count_zscore_2"]
        )
        pd.testing.assert_series_equal(
            single["trade_count_zscore_3"], batch["trade_count_zscore_3"]
        )

    def test_custom_column_name_fn(self):
        df = _ohlcv_tc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 1, 10),
                ("2025-01-02 09:32", 100, 101, 99, 100, 1, 20),
            ],
        )
        result = add_feature_trade_count_zscore(
            df, [2], column_name_fn=lambda w: f"tcz_{w}"
        )
        self.assertIn("tcz_2", result.columns)
        self.assertNotIn("trade_count_zscore_2", result.columns)

    def test_window_zero_raises(self):
        df = _ohlcv_tc(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 100, 1, 100)],
        )
        with self.assertRaises(ValueError):
            add_feature_trade_count_zscore(df, [0])

    def test_combined_matches_separate_calls(self):
        df = _ohlcv_tc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 1, 10),
                ("2025-01-02 09:32", 100, 101, 99, 100, 1, 20),
                ("2025-01-02 09:33", 100, 101, 99, 100, 1, 30),
            ],
        )
        sep = add_feature_volume_zscore(df.copy(), [2, 3])
        sep = add_feature_trade_count_zscore(sep, [2, 3])
        both = add_feature_volume_and_trade_count_zscore(df.copy(), [2, 3])
        for col in ["volume_zscore_2", "volume_zscore_3", "trade_count_zscore_2", "trade_count_zscore_3"]:
            pd.testing.assert_series_equal(sep[col], both[col])


class TestAddFeatureDayOfWeek(unittest.TestCase):
    """Tests for add_feature_day_of_week (Monday=1 .. Sunday=7)."""

    def test_monday_is_one(self):
        # 2025-01-06 is Monday (US).
        df = _ohlc(
            "AAPL",
            [("2025-01-06 09:31", 100, 101, 99, 100)],
        )
        result = add_feature_day_of_week(df)
        self.assertIn("day_of_week", result.columns)
        self.assertEqual(int(result["day_of_week"].iloc[0]), 1)

    def test_sunday_is_seven(self):
        # 2025-01-05 is Sunday.
        df = _ohlc(
            "AAPL",
            [("2025-01-05 10:00", 100, 101, 99, 100)],
        )
        result = add_feature_day_of_week(df)
        self.assertEqual(int(result["day_of_week"].iloc[0]), 7)

    def test_tz_aware_uses_new_york_calendar(self):
        # 2025-01-06 04:00 UTC = 2025-01-05 23:00 EST -> still Sunday in NY -> 7.
        ts = pd.Timestamp("2025-01-06 04:00:00", tz="UTC")
        index = pd.MultiIndex.from_arrays([["AAPL"], [ts]], names=["symbol", "timestamp"])
        df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]},
            index=index,
        )
        result = add_feature_day_of_week(df)
        self.assertEqual(int(result["day_of_week"].iloc[0]), 7)

    def test_multiple_rows(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-06 09:31", 100, 101, 99, 100),  # Mon
                ("2025-01-07 09:31", 100, 101, 99, 100),  # Tue
            ],
        )
        result = add_feature_day_of_week(df)
        self.assertEqual(int(result["day_of_week"].iloc[0]), 1)
        self.assertEqual(int(result["day_of_week"].iloc[1]), 2)

    def test_custom_column_name(self):
        df = _ohlc("AAPL", [("2025-01-06 09:31", 100, 101, 99, 100)])
        result = add_feature_day_of_week(df, column_name="dow")
        self.assertIn("dow", result.columns)
        self.assertNotIn("day_of_week", result.columns)
        self.assertEqual(int(result["dow"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
