# Backtest

Run strategies over historical bars using the framework (orders, portfolio, strategy, data feed, broker, clock). Data shape matches `lib.stock`: MultiIndex `(symbol, timestamp)` with OHLCV columns.

## Entry point

Use **`run(data, strategy, initial_cash, record_equity_curve=True)`** from `lib.backtest`. It builds a DataFrame-backed data feed, sim clock, and sim broker, runs the strategy bar-by-bar, applies fills to the portfolio, and returns a **`BacktestResult`** with `portfolio` and `equity_curve` (list of `(datetime, float)`).

## Example

```python
import pandas as pd
from lib.backtest import run
from lib.framework import Order, OrderSide, Portfolio


class BuyAndHold:
    """Buy 10 shares on the first bar and hold."""

    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol

    def next(self, current_time, market_snapshot: pd.DataFrame, portfolio: Portfolio):
        if portfolio.position(self.symbol).quantity == 0 and not market_snapshot.empty:
            return [Order(symbol=self.symbol, side=OrderSide.BUY, qty=10)]
        return []


# Data: MultiIndex (symbol, timestamp) with open, high, low, close, volume
# (e.g. from lib.stock.StockDataFetcher.get_historical_bars + cleaner)
data = ...  # pd.DataFrame with required index and columns

strategy = BuyAndHold("AAPL")
result = run(data, strategy, initial_cash=50_000.0)

# Inspect result
print(result.portfolio.cash)
print(result.portfolio.position("AAPL").quantity)
print(len(result.portfolio.trade_history))
print(result.equity_curve[-1])  # (last_time, last_equity)
```

## Modules

- **engine** — `run()` and `BacktestResult`.
- **data_feed** — `DataFrameDataFeed(data)`: implements framework `DataFeed`; `get_bars(current_time, symbol=None)` returns the bar(s) at that time.
- **sim_clock** — `SimClock(timestamps)`: advances bar-by-bar; implements framework `Clock` and adds `advance()`.
- **sim_broker** — `SimBroker()`: implements framework `Broker`; fills market orders at bar close. Call `set_current_bars(snapshot, time)` before `get_fills()` (the engine does this).
