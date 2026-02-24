"""
Unit tests for Strategy protocol.
"""

import unittest
from datetime import datetime, timezone

import pandas as pd

from lib.framework import Order, Portfolio, Strategy


def _ts(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, 14, 30, 0, tzinfo=timezone.utc)


class DummyStrategy:
    """Minimal strategy that returns fixed orders for protocol testing."""

    def __init__(self, orders_to_return: list[Order]):
        self.orders_to_return = orders_to_return

    def next(
        self,
        current_time: datetime,
        market_snapshot: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[Order]:
        return list(self.orders_to_return)


class TestStrategyProtocol(unittest.TestCase):
    """Test that a concrete strategy satisfies the Strategy protocol."""

    def test_dummy_strategy_returns_orders(self):
        """Strategy.next returns the list of orders."""
        orders = [Order(symbol="AAPL", side="buy", qty=10)]
        strategy: Strategy = DummyStrategy(orders_to_return=orders)
        snapshot = pd.DataFrame({"close": [150.0]})
        portfolio = Portfolio(cash=10_000.0)
        out = strategy.next(_ts(2024, 1, 15), snapshot, portfolio)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].symbol, "AAPL")
        self.assertEqual(out[0].qty, 10)

    def test_dummy_strategy_can_return_empty(self):
        """Strategy.next can return empty list."""
        strategy: Strategy = DummyStrategy(orders_to_return=[])
        snapshot = pd.DataFrame({"close": [150.0]})
        portfolio = Portfolio(cash=10_000.0)
        out = strategy.next(_ts(2024, 1, 15), snapshot, portfolio)
        self.assertEqual(out, [])
