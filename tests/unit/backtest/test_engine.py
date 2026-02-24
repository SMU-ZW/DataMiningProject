"""
Unit tests for backtest engine: synthetic bars and dummy strategy.
"""

import unittest
from datetime import datetime, timezone

import pandas as pd

from lib.backtest.engine import BacktestResult, run
from lib.framework import Order, OrderSide, Portfolio


def _ts(s: str) -> datetime:
    return pd.Timestamp(s, tz="UTC").to_pydatetime()


def _make_bars(symbol: str, times: list[str], close_prices: list[float]) -> pd.DataFrame:
    """Build a minimal MultiIndex DataFrame (symbol, timestamp) with close column."""
    n = len(times)
    index = pd.MultiIndex.from_arrays(
        [[symbol] * n, pd.to_datetime(times, utc=True)],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": close_prices, "high": close_prices, "low": close_prices, "close": close_prices, "volume": [1000] * n},
        index=index,
    )


class BuyOnceStrategy:
    """Strategy that buys 10 shares on the first bar only (when no position yet)."""

    def next(self, current_time, market_snapshot, portfolio):
        if (
            portfolio.position("AAPL").quantity == 0
            and portfolio.cash >= 1000
            and market_snapshot is not None
            and not market_snapshot.empty
        ):
            return [Order(symbol="AAPL", side=OrderSide.BUY, qty=10)]
        return []


class TestBacktestEngine(unittest.TestCase):
    """Test backtest run() with synthetic data and dummy strategy."""

    def test_run_buy_once_strategy(self):
        """Engine runs strategy; buy order is filled; portfolio and fills are correct."""
        data = _make_bars("AAPL", ["2024-01-02 14:30:00", "2024-01-03 14:30:00"], [100.0, 102.0])
        strategy = BuyOnceStrategy()
        result = run(data, strategy, initial_cash=10_000.0, record_equity_curve=True)
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.portfolio, Portfolio)
        self.assertEqual(result.portfolio.cash, 10_000.0 - 1000.0)
        pos = result.portfolio.position("AAPL")
        self.assertEqual(pos.quantity, 10)
        self.assertEqual(len(result.portfolio.trade_history), 1)
        self.assertEqual(result.portfolio.trade_history[0].qty, 10)
        self.assertEqual(result.portfolio.trade_history[0].price, 100.0)
        self.assertEqual(len(result.equity_curve), 2)

    def test_run_empty_data(self):
        """Run with empty DataFrame completes without error; portfolio unchanged."""
        data = pd.DataFrame()
        strategy = BuyOnceStrategy()
        result = run(data, strategy, initial_cash=5_000.0, record_equity_curve=False)
        self.assertEqual(result.portfolio.cash, 5_000.0)
        self.assertEqual(len(result.portfolio.positions), 0)
        self.assertEqual(len(result.equity_curve), 0)
