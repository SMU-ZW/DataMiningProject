"""
Clock abstraction: current time and advance.
Backtest advances bar-by-bar; live uses real time.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class Clock(ABC):
    """Abstract clock: provides current time."""

    @property
    @abstractmethod
    def current_time(self) -> datetime:
        """Current time (bar time in backtest, now in live)."""
