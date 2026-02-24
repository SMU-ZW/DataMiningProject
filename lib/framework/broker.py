"""
Broker/execution abstraction: submit orders, get fills.
Backtest implementation simulates fills at bar close; paper/live will call Alpaca (or other).
"""

from abc import ABC, abstractmethod
from typing import List

from lib.framework.orders import Fill, Order


class Broker(ABC):
    """Abstract broker: submit orders and receive fills (e.g. poll get_fills after each step)."""

    @abstractmethod
    def submit(self, order: Order) -> None:
        """Submit an order for execution."""

    @abstractmethod
    def get_fills(self) -> List[Fill]:
        """Return fills since last call (or since submit). Call after each step to process fills."""
