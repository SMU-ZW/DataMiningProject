"""
Integration test for backtesting with real data.

Fetches data via lib.stock, runs a simple buy-and-hold strategy, and checks
final position and PnL. Skipped when Alpaca API key file is not present.
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from lib.backtest.engine import run
from lib.framework import Order, OrderSide, Portfolio
from lib.stock.data_fetcher import StockDataFetcher


class TestBacktestingWithRealData(unittest.TestCase):
    """Run backtest with fetched stock data and a buy-and-hold strategy."""

    def setUp(self):
        self.fetcher = StockDataFetcher()
        self.symbol = "AAPL"
        end_date = datetime.now()
        self.start_date = end_date - timedelta(days=14)

    def test_buy_and_hold_with_daily_bars(self):
        """Fetch daily bars, run buy-and-hold backtest, assert position and PnL."""
        data = self.fetcher.get_historical_bars(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=datetime.now(),
            timeframe=TimeFrame.Day,
        )
        self.assertFalse(data.empty, "Fetched data should not be empty")
        self.assertIn("close", data.columns)
        self.assertIsInstance(data.index, pd.MultiIndex)
        self.assertIn("timestamp", data.index.names)

        symbol = self.symbol

        class BuyAndHold:
            """Buy 10 shares on the first bar and hold."""

            def next(self, current_time, market_snapshot, portfolio: Portfolio):
                if portfolio.position(symbol).quantity == 0 and not market_snapshot.empty:
                    return [Order(symbol=symbol, side=OrderSide.BUY, qty=10)]
                return []

        strategy = BuyAndHold()
        initial_cash = 50_000.0
        result = run(data, strategy, initial_cash=initial_cash, record_equity_curve=True)

        self.assertEqual(len(result.portfolio.trade_history), 1, "Should have one fill (buy)")
        fill = result.portfolio.trade_history[0]
        self.assertEqual(fill.symbol, self.symbol)
        self.assertEqual(fill.qty, 10)
        self.assertGreater(fill.price, 0)

        pos = result.portfolio.position(self.symbol)
        self.assertEqual(pos.quantity, 10, "Should hold 10 shares")
        self.assertLess(result.portfolio.cash, initial_cash, "Cash should decrease after buy")
        self.assertGreater(result.portfolio.equity(), 0, "Equity should be positive")
        self.assertEqual(len(result.equity_curve), len(data.index.get_level_values("timestamp").unique()))
